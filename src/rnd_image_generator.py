import os, io
import time
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import random
import json

from src.converters import JsonConverter
from src.quickdraw.parse_qd import ShardedTFRecordConverter

MAX_LINES = 20

def create_image(width=800, height=600, num_of_images=100):
    width = int(width)
    height = int(height)
    num_of_images = int(num_of_images)

    current = time.strftime("%Y%m%d%H%M%S")
    os.mkdir(current)

    for n in range(num_of_images):
        filename = '{0}/{0}_{1:03d}.jpg'.format(current, n)
        rgb_array = np.random.rand(height, width, 3) * 255
        image = Image.fromarray(rgb_array.astype('uint8')).convert('RGB')
        image.save(filename)


def create_gray_img(input):
    face = np.load(input)[0]
    base_img = np.zeros((100, 100, 1))


class ColorGenerator:

    @staticmethod
    def get_line_color():
        return 0

    @staticmethod
    def get_curve_color():
        return 0


class RGBGenerator(ColorGenerator):

    @staticmethod
    def get_line_color():
        return random.randint(100, 200), 0, random.randint(0, 100)

    @staticmethod
    def get_curve_color():
        return 0, random.randint(100, 200), random.randint(100, 200)


class GrayGenerator(ColorGenerator):

    @staticmethod
    def get_line_color():
        return random.randint(100, 225)

    @staticmethod
    def get_curve_color():
        return random.randint(50, 180)


class LineGenerator:

    def __init__(self, is_colored=False):
        self.is_colored = is_colored
        if self.is_colored:
            self.colorGenerator = RGBGenerator()
        else:
            self.colorGenerator = GrayGenerator()

    def generate_curve(self, drawer, image_size):
        pass


class PolylineGenerator(LineGenerator):

    def generate_curve(self, drawer, image_size):
        x0, y0 = random.randint(0, image_size[0]), random.randint(0, image_size[1])
        x1, y1 = random.randint(0, image_size[0]), random.randint(0, image_size[1])
        x2, y2 = random.randint(x1, image_size[0]), random.randint(y1, image_size[1])
        drawer.line((x0, y0) + (x1, y1), fill = self.colorGenerator.get_line_color())
        drawer.line((x1, y1) + (x2, y2), fill = self.colorGenerator.get_line_color())


class CurveGenerator(LineGenerator):

    def generate_curve(self, drawer, image_size):
        x0, y0 = random.randint(0, image_size[0]), random.randint(0, image_size[1])
        x1, y1 = random.randint(x0, image_size[0]), random.randint(y0, image_size[1])
        start = random.randint(0, 270)
        end = min(start + random.randint(30, 360), 360)
        drawer.arc((x0, y0) + (x1, y1), start, end, fill = self.colorGenerator.get_curve_color())


class ImageGenerator:

    def __init__(self, is_colored=False):
        self.polyGenerator = PolylineGenerator(is_colored)
        self.curveGenerator = CurveGenerator(is_colored)
        if is_colored:
            self.image_type = "RGB"
        else:
            self.image_type = "L"

    def generate_image(self, id, width=100, height=100, save=False):
        im = Image.new(self.image_type, (width, height))
        draw = ImageDraw.Draw(im)

        q = random.randint(0, MAX_LINES)

        for x in range(MAX_LINES):
            if x < q:
                self.polyGenerator.generate_curve(draw, im.size)
            else :
                self.curveGenerator.generate_curve(draw, im.size)

        if save:
            filename = "train_img_{}_q{}.png".format(id, q)
            print("Saving " + filename)
            im.save("img_gen/" + filename)
        return im, id


BASE_IMAGE_SIZE = 100
OBJECT_SIZE = 28
LABEL = 'face'


def pil_image_to_bytes(pil_image):
    imgByteArr = io.BytesIO()
    pil_image.save(imgByteArr, format='JPEG')
    return imgByteArr.getvalue()


def place_face(base_img, face):
    pixels = base_img.load()
    obj_x = np.random.randint(0, BASE_IMAGE_SIZE - OBJECT_SIZE)
    obj_y = np.random.randint(0, BASE_IMAGE_SIZE - OBJECT_SIZE)
    for i, px in enumerate(face):
        col = int(i / OBJECT_SIZE)
        row = i % OBJECT_SIZE
        pixels[row + obj_y, col + obj_x] = int(px)
    return base_img, obj_x / BASE_IMAGE_SIZE, obj_y / BASE_IMAGE_SIZE


def load_img(face, image_id, converter, is_test=False):
    img_generator = ImageGenerator()
    generated, img_id = img_generator.generate_image(image_id, save=False)
    generated, bb_box_x, bb_box_y = place_face(generated, face)
    # output_file='../data/faces/train_img_face_{}.jpg'.format(img_id)
    # generated.save(output_file)
    rgbimg = Image.new("RGB", generated.size)
    rgbimg.paste(generated)
    if is_test:
        rgbimg.save('../data/faces/test/test_face_{}.jpg'.format(img_id))
        return
    # s = rgbimg.tobytes().decode("latin1")
    result = {
        "id":str(img_id),
        "category": "face",
        "bb_box_xmin": bb_box_x,
        "bb_box_xmax": bb_box_x + OBJECT_SIZE / BASE_IMAGE_SIZE,
        "bb_box_ymin": bb_box_y,
        "bb_box_ymax": bb_box_y + OBJECT_SIZE / BASE_IMAGE_SIZE,
        "width": BASE_IMAGE_SIZE,
        "height": BASE_IMAGE_SIZE,
        "img":pil_image_to_bytes(rgbimg)
    }
    converter.convert_sharded(result, img_id)
    # with open(output_file, "a") as file:
    #     json.dump(result, file)
    #     print('', file = file)


def prepare_dataset():
    train_size = 10000
    validation_size = 1000
    test_size = 100

    images = np.load('../data/quick_draw/full_numpy_bitmap_face.npy')
    rng = np.random.default_rng()
    total_size = images.size

    total_generation_size = train_size + validation_size + test_size

    converter = JsonConverter()

    # train_indexes = rng.choice(images[:train_size], train_size)
    sharded_converter = ShardedTFRecordConverter('../data/faces/{}'.format("train"), LABEL, converter)
    for i in range(train_size):
        load_img(images[i], i, sharded_converter)
        if i % 100 == 0:
            print("Progress: {}".format(i / total_generation_size))

    sharded_converter = ShardedTFRecordConverter('../data/faces/{}'.format("validation"), LABEL, converter)
    for i in range(validation_size):
        load_img(images[i + train_size], i, sharded_converter)
        if (i + train_size) % 100 == 0:
            print("Progress: {}".format((i + train_size) / total_generation_size))

    for i in range(test_size):
        load_img(images[i + train_size + validation_size], i, sharded_converter, is_test=True)
        if (i + train_size + validation_size) % 100 == 0:
            print("Progress: {}".format((i + train_size + validation_size) / total_generation_size))


def plot_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()


prepare_dataset()
# make_tf_records()


