import numpy as np

def normalize(input_img):
    img_min = input_img.min()
    img_max = input_img.max()
    x = (input_img - img_min) / (img_max - img_min)
    x[np.isnan(x)] = 0
    return x
