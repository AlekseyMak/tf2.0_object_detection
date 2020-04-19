import os, io
import time
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import random
import json


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
        return im
