from enum import Enum

from src.generation.converters import JsonConverter
from src.generation.bg_generator import ImageGenerator
from src.generation.img_utils import *
from src.generation.parse_qd import ShardedTFRecordConverter


BASE_IMAGE_SIZE = 100
OBJECT_SIZE = 28
LABEL = 'face'


class BBox:

    #Contains normalised coords
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax


class AugmentMode(Enum):
    NONE = 0
    UPSCALE = 1
    MIRROR_H = 2
    ROTATE_CW = 3
    ROTATE_CCW = 4


def augment_face(base_face, mode):
    return {
        AugmentMode.NONE: np_to_pil(base_face),
        AugmentMode.UPSCALE: resize_np_img(base_face, OBJECT_SIZE * 2, OBJECT_SIZE * 2),
        AugmentMode.MIRROR_H: np_to_pil(np.fliplr(base_face.reshape((28,28)))),
        AugmentMode.ROTATE_CW: rotate_np_img(base_face, 30),
        AugmentMode.ROTATE_CCW: rotate_np_img(base_face, -30)
    }.get(mode, np_to_pil(base_face))


def place_single_face(base_img, face, mode):
    augmented_face = augment_face(face, mode)  # returns PIL
    obj_x = np.random.randint(0, BASE_IMAGE_SIZE - augmented_face.width)
    obj_y = np.random.randint(0, BASE_IMAGE_SIZE - augmented_face.height)
    new_img = base_img.copy()
    new_img.paste(augmented_face, (obj_x, obj_y))
    box = BBox(
        xmin=obj_x / BASE_IMAGE_SIZE,
        xmax=(obj_x + augmented_face.width) / BASE_IMAGE_SIZE,
        ymin=obj_y / BASE_IMAGE_SIZE,
        ymax=(obj_y + augmented_face.height) / BASE_IMAGE_SIZE
    )
    entry = {
        'img': new_img,
        'box': box
    }
    return entry


# base_img: PIL image
# face: np array
#
# returns batch of augmented images
#
def place_faces(base_img, face):
    batch = []
    # for mode in AugmentMode.NONE:
    entry = place_single_face(base_img, face, AugmentMode.NONE)
    batch.append(entry)

    return batch


def generate_img(face, image_id, converter, is_test=False, draw_box=False):
    img_generator = ImageGenerator()
    background = img_generator.generate_image(id=image_id,
                                             width=BASE_IMAGE_SIZE,
                                             height=BASE_IMAGE_SIZE,
                                             save=False)
    faced_images = place_faces(background, face)
    # output_file='../data/faces/train_img_face_{}.jpg'.format(img_id)
    # generated.save(output_file)
    for index, entry in enumerate(faced_images):
        box = entry['box']
        rgbimg = Image.new("RGB", background.size)
        rgbimg.paste(entry['img'])
        if is_test:
            rgbimg.save('../data/faces/test/test_face_{}_{}.jpg'.format(image_id, index))

        if draw_box:
            draw = ImageDraw.Draw(rgbimg)
            draw.line(
                [(box.xmin * BASE_IMAGE_SIZE, box.ymin * BASE_IMAGE_SIZE),
                 (box.xmin * BASE_IMAGE_SIZE, box.ymax * BASE_IMAGE_SIZE),
                 (box.xmax * BASE_IMAGE_SIZE, box.ymax * BASE_IMAGE_SIZE),
                 (box.xmax * BASE_IMAGE_SIZE, box.ymin * BASE_IMAGE_SIZE),
                 (box.xmin * BASE_IMAGE_SIZE, box.ymin * BASE_IMAGE_SIZE)],
                width=1,
                fill='yellow')
            rgbimg.save('../data/faces/test/test_face_{}_{}_box.jpg'.format(image_id, index))

        result = {
            "id": "{}_{}".format(image_id, index),
            "category": "face",
            "bb_box_xmin": box.xmin,
            "bb_box_xmax": box.xmax,
            "bb_box_ymin": box.ymin,
            "bb_box_ymax": box.ymax,
            "width": BASE_IMAGE_SIZE,
            "height": BASE_IMAGE_SIZE,
            "img":pil_image_to_bytes(rgbimg)
        }

        if converter != None:
            converter.convert_sharded(result, image_id * len(AugmentMode) + index)
    # with open(output_file, "a") as file:
    #     json.dump(result, file)
    #     print('', file = file)


def prepare_dataset():
    train_size = 10000
    validation_size = 2000
    test_size = 10
    skip = 0

    images = np.load('../data/quick_draw/full_numpy_bitmap_face.npy')
    rng = np.random.default_rng()
    total_size = images.size

    total_generation_size = train_size + validation_size + test_size

    converter = JsonConverter()

    # train_indexes = rng.choice(images[:train_size], train_size)
    if train_size > 0:
        sharded_converter = ShardedTFRecordConverter('../data/faces/{}'.format("train"), LABEL, converter)
        for i in range(train_size):
            generate_img(images[i], i, sharded_converter)
            if i % 100 == 0:
                print("Progress: {}".format(i / total_generation_size))

    if validation_size > 0:
        sharded_converter = ShardedTFRecordConverter('../data/faces/{}'.format("validation"), LABEL, converter)
        for i in range(validation_size):
            generate_img(images[i + train_size], i, sharded_converter)
            if (i + train_size) % 100 == 0:
                print("Progress: {}".format((i + train_size) / total_generation_size))

    for i in range(test_size):
        generate_img(images[skip + i + train_size + validation_size], i, None, is_test=True, draw_box=True)
        if (i + train_size + validation_size) % 100 == 0:
            print("Progress: {}".format((i + train_size + validation_size) / total_generation_size))



prepare_dataset()



