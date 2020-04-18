import os, io
import time
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import random
import json


def pil_image_to_bytes(pil_image):
    imgByteArr = io.BytesIO()
    pil_image.save(imgByteArr, format='JPEG')
    return imgByteArr.getvalue()


def plot_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def resize_img(img, width=70, height=70):
    return img.resize((width, height), resample=Image.BOX)