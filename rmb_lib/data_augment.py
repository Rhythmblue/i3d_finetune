from __future__ import division
import random
from PIL import ImageOps


def transform_data(data, crop_size=224, will_crop=True, will_flip=False):
    length = data[0].size[0]
    width = data[0].size[1]
    if will_crop:
        x0 = random.randint(0, length - crop_size)
        y0 = random.randint(0, width - crop_size)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i, img in enumerate(data):
            data[i] = img.crop((x0, y0, x1, y1))
    if will_flip & x0%2 == 0:
        for i, img in enumerate(data):
            data[i] = ImageOps.mirror(img)
    return  data
