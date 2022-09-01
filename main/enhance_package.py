import yaml
import numpy as np
from torch.nn import Module
from torchvision.transforms import *

from image_enhance.random_flip import RandomFlip
from image_enhance.letter_box import LetterBox
from image_enhance.augment_hsv import RandomHSV
from image_enhance.random_perspective import RandomPerspective


class EnhancePackage(Module):
    def __init__(self, target_shape, enhance_cfg):
        super(EnhancePackage, self).__init__()

        cfg = enhance_cfg
        random_perspective = RandomPerspective(cfg["degrees"],
                                               cfg["translate"],
                                               cfg["scale"],
                                               cfg["shear"],
                                               cfg["perspective"])
        random_hsv = RandomHSV(cfg["hsv_h"],
                               cfg["hsv_s"],
                               cfg["hsv_v"])
        random_flip = RandomFlip(cfg["flip_ud"],
                                 cfg["flip_lr"])
        self.letter_box = LetterBox(target_shape,
                                    cfg["scale_fill"])

        random_equalize = RandomEqualize(cfg["equalize"])

        self.enhance_without_label = [random_equalize]
        # Todo Remove Random Perspective
        self.enhance = [random_hsv,
                        random_flip, ]

    def __call__(self, image, labels, enhance=True):

        if enhance:
            for e in self.enhance_without_label:
                image = e(image)
            # image = image.numpy()
            for e in self.enhance:
                image = np.ascontiguousarray(image)
                image, labels = e(image, labels)

        image, labels = self.letter_box(image, labels)
        return image, labels

    @staticmethod
    def get_dataset_cfg(cfg_file):
        with open(cfg_file, 'r') as file:
            cfg = yaml.safe_load(file)
            return cfg


if __name__ == "__main__":
    from utils.bbox import *
    from utils.helper_cv import *
    import numpy as np

    test_image_file = r"F:\PASCALVOC\VOC2012\JPEGImages\2007_000733.jpg"
    test_image = np.asarray(cv2.imread(test_image_file, flags=cv2.IMREAD_COLOR))  # [H,W,C]
    test_bbox = np.asarray([[48, 25, 273, 383], [103, 201, 448, 435]])  # mode=x1y1x2y2
    _cfg = EnhancePackage.get_dataset_cfg("../cfg/enhance/enhance.yaml")
    enhanceImage, __bbox = EnhancePackage(640, _cfg)(test_image, test_bbox, enhance=True)  # [C,H,W]
    print(enhanceImage.shape)
    show_bbox(enhanceImage, __bbox)
