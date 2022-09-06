import colorsys

import cv2
import numpy as np
from numpy import ndarray
from torch import Tensor

import utils.bbox
from utils.bbox import BBoxType, CvtFlag


def cvt_for_transform(image: Tensor):
    """
    KeyPoint Cv2 Image: [H,W,C] => Transform: [C,H,W]
    :param image:
    """
    return image.permute(2, 0, 1)


def cvt_for_cv(image: Tensor):
    """
    KeyPoint Transform: [C,H,W]  => Cv2 Image: [H,W,C]
    :rtype: Tensor
    :param image: Tensor
    """
    return image.permute(1, 2, 0)


# 保持原图比例将长边缩放到target_size
def resize_by_largeborder(img, target_size):
    largeborder = max(img.shape)
    imgh, imgw = img.shape[:2]
    scalefactor = target_size / largeborder
    newh = int(imgh * scalefactor)
    neww = int(imgw * scalefactor)
    img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
    return img


def resize_and_padding(image, new_shape):
    """
        image : (h, w) or (h, w, c)
        new_shape : (h, w)
    """
    new_shape = tuple(new_shape)
    imgh, imgw = image.shape[:2]
    h, w = new_shape
    scalefactor = np.min((w / imgw, h / imgh))
    neww = int(imgw * scalefactor)
    newh = int(imgh * scalefactor)
    if image.ndim == 2:
        new_image = np.zeros(new_shape, image.dtype)
    else:
        new_image = np.zeros(new_shape + (image.shape[2],), image.dtype)
    offseth, offsetw = (h - newh) // 2, (w - neww) // 2
    image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_NEAREST)
    new_image[offseth:offseth + newh, offsetw:offsetw + neww] = image


def generate_colors(color_count):
    hsv_tuples = [(x / color_count, 1., 1.) for x in range(color_count)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors



