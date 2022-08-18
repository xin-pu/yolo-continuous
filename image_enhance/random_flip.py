import numpy as np
from random import Random

from numpy import ndarray
from torch.nn import Module


class RandomFlip(Module):
    def __init__(self, flip_lr_prod=0.5, flip_ud_prob=0.5):
        """
        随机翻转
        :param flip_lr_prod: 垂直翻转概率
        :param flip_ud_prob: 水平翻转概率
        """
        super(RandomFlip, self).__init__()
        self.flip_lr_prod = flip_lr_prod
        self.flip_ud_prob = flip_ud_prob

    def __call__(self, image: ndarray, bbox_xyxy):
        ran = Random()
        h, w = image.shape[0], image.shape[1]
        _image = image
        _bbox = bbox_xyxy
        if ran.random() < self.flip_lr_prod:  # 水平翻转
            _image = np.fliplr(_image)
            x_max = w - _bbox[:, 0]
            x_min = w - _bbox[:, 2]
            _bbox[:, 0] = x_min
            _bbox[:, 2] = x_max

        if ran.random() < self.flip_ud_prob:  # 垂直翻转
            _image = np.flipud(_image)
            y_max = h - _bbox[:, 1]
            y_min = h - _bbox[:, 3]
            _bbox[:, 1] = y_min
            _bbox[:, 3] = y_max

        return _image, _bbox


if __name__ == "__main__":
    from utils.bbox import *
    from utils.helper_cv import *

    test_image_file = r"F:\PASCALVOC\VOC2012\JPEGImages\2007_000733.jpg"

    test_image = np.asarray(cv2.imread(test_image_file, flags=cv2.IMREAD_COLOR))  # [H,W,C]
    test_bbox = np.asarray([[48, 25, 273, 383], [103, 201, 448, 435]])  # mode=xyxy

    enhanceImage, __bbox = RandomFlip(flip_lr_prod=0.5, flip_ud_prob=0.5)(test_image, test_bbox)  # [H,W,C]
    print(__bbox)
    show_bbox(enhanceImage, __bbox)
