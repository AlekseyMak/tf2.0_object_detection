import os, io
import time
from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import random
import json

from src.converters import JsonConverter
from src.generation.bg_generator import ImageGenerator
from src.generation.img_utils import pil_image_to_bytes
from src.quickdraw.parse_qd import ShardedTFRecordConverter


BASE_IMAGE_SIZE = 400
OBJECT_SIZE = 28
LABEL = 'face'


class AugmentMode(Enum):
    UPSCALE = 1
    MIRROR_H = 2
    ROTATE_CW = 3
    ROTATE_CCW = 4

def augment_face(base_face, mode):
    return {
        AugmentMode.UPSCALE:
            augmented_face = resize_img(base_face)
    }

def place_faces(base_img, face):
    pixels = base_img.load()
    obj_x = np.random.randint(0, BASE_IMAGE_SIZE - OBJECT_SIZE)
    obj_y = np.random.randint(0, BASE_IMAGE_SIZE - OBJECT_SIZE)
    for i, px in enumerate(face):
        col = int(i / OBJECT_SIZE)
        row = i % OBJECT_SIZE
        pixels[row + obj_y, col + obj_x] = int(px)
    return base_img, obj_x / BASE_IMAGE_SIZE, obj_y / BASE_IMAGE_SIZE


def generate_img(face, image_id, converter, is_test=False):
    img_generator = ImageGenerator()
    background = img_generator.generate_image(id=image_id,
                                             width=BASE_IMAGE_SIZE,
                                             height=BASE_IMAGE_SIZE,
                                             save=False)
    faced_images, bb_box_x, bb_box_y = place_faces(background, face)
    # output_file='../data/faces/train_img_face_{}.jpg'.format(img_id)
    # generated.save(output_file)
    rgbimg = Image.new("RGB", background.size)
    rgbimg.paste(background)
    if is_test:
        rgbimg.save('../data/faces/test/test_face_{}.jpg'.format(image_id))
        return
    # s = rgbimg.tobytes().decode("latin1")
    result = {
        "id":str(image_id),
        "category": "face",
        "bb_box_xmin": bb_box_x,
        "bb_box_xmax": bb_box_x + OBJECT_SIZE / BASE_IMAGE_SIZE,
        "bb_box_ymin": bb_box_y,
        "bb_box_ymax": bb_box_y + OBJECT_SIZE / BASE_IMAGE_SIZE,
        "width": BASE_IMAGE_SIZE,
        "height": BASE_IMAGE_SIZE,
        "img":pil_image_to_bytes(rgbimg)
    }
    converter.convert_sharded(result, image_id)
    # with open(output_file, "a") as file:
    #     json.dump(result, file)
    #     print('', file = file)


def prepare_dataset():
    train_size = 10000
    validation_size = 2000
    test_size = 20

    images = np.load('../data/quick_draw/full_numpy_bitmap_face.npy')
    rng = np.random.default_rng()
    total_size = images.size

    total_generation_size = train_size + validation_size + test_size

    converter = JsonConverter()

    # train_indexes = rng.choice(images[:train_size], train_size)
    sharded_converter = ShardedTFRecordConverter('../data/faces/{}'.format("train"), LABEL, converter)
    for i in range(train_size):
        generate_img(images[i], i, sharded_converter)
        if i % 100 == 0:
            print("Progress: {}".format(i / total_generation_size))

    sharded_converter = ShardedTFRecordConverter('../data/faces/{}'.format("validation"), LABEL, converter)
    for i in range(validation_size):
        generate_img(images[i + train_size], i, sharded_converter)
        if (i + train_size) % 100 == 0:
            print("Progress: {}".format((i + train_size) / total_generation_size))

    for i in range(test_size):
        generate_img(images[i + train_size + validation_size], i, sharded_converter, is_test=True)
        if (i + train_size + validation_size) % 100 == 0:
            print("Progress: {}".format((i + train_size + validation_size) / total_generation_size))





# prepare_dataset()
# make_tf_records()

# from PIL import Image



