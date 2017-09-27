import random


def transform_data(data, crop_size=224, will_crop=True):
    length = data[0].size[0]
    width = data[0].size[1]
    if will_crop:
        x0 = random.randint(0, length - crop_size)
        y0 = random.randint(0, width - crop_size)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i in range(len(data)):
            data[i] = data[i].crop((x0, y0, x1, y1))

    return  data
