import torch

from random import Random
from torch.nn import Module
from torchvision.transforms import *


class RandomFlip(Module):
    def __init__(self, flip_lr_prod=0.5, flip_ud_prob=0.5):
        super(RandomFlip, self).__init__()
        self.flip_lr_prod = flip_lr_prod
        self.flip_ud_prob = flip_ud_prob

    def __call__(self, image, bbox):
        ran = Random()
        h, w = image.shape[1], image.shape[2]
        _image = image.clone()
        _bbox = bbox.clone()
        if ran.random() <= self.flip_lr_prod:  # 水平翻转
            _image = RandomHorizontalFlip(p=1)(_image)
            x_max = w - _bbox[:, 0]
            x_min = w - _bbox[:, 2]
            _bbox[:, 0] = x_min
            _bbox[:, 2] = x_max

        if ran.random() <= self.flip_ud_prob:  # 垂直翻转
            _image = RandomVerticalFlip(p=1)(_image)
            y_max = h - _bbox[:, 1]
            y_min = h - _bbox[:, 3]
            _bbox[:, 1] = y_min
            _bbox[:, 3] = y_max

        return _image, _bbox


if __name__ == "__main__":
    from utils.bbox import *
    from utils.helper_cv import *

    test_image_file = r"F:\PASCALVOC\VOC2012\JPEGImages\2007_000733.jpg"
    test_image = torch.asarray(cv2.imread(test_image_file, flags=cv2.IMREAD_COLOR))  # [H,W,C]
    test_image = cvt_for_transform(test_image)
    test_bbox = torch.asarray([[48, 25, 273, 383], [103, 201, 448, 435]])  # mode=xyxy

    enhanceImage, __bbox = RandomFlip(flip_lr_prod=0.5, flip_ud_prob=0.5)(test_image, test_bbox)  # [C,H,W]
    print(__bbox)
    show_bbox(cvt_for_cv(enhanceImage), __bbox)
