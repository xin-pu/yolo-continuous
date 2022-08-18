import math
import cv2
import numpy as np
from random import Random

from numpy import ndarray
from torch.nn import Module
from torchvision.transforms import *


class RandomPerspective(Module):
    def __init__(self,
                 degrees=10,
                 translate=.1,
                 scale=.1,
                 shear=10,
                 perspective=0.0,
                 border=(0, 0)):

        super(RandomPerspective, self).__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border

    def __call__(self, img: ndarray, target_xyxy):
        degrees = self.degrees
        translate = self.translate
        scale = self.scale
        shear = self.shear
        perspective = self.perspective
        border = self.border

        random = Random()
        height = img.shape[0] + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + border[1] * 2

        # Center
        center = np.eye(3)
        center[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        center[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        pers = np.eye(3)
        pers[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        pers[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        rotation = np.eye(3)
        a = random.uniform(-degrees, degrees)
        s = random.uniform(1 - scale, 1.1 + scale)
        rotation[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        _shear = np.eye(3)
        _shear[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        _shear[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation 平移
        translation = np.eye(3)
        translation[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        translation[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        matrix = translation @ _shear @ rotation @ pers @ center  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (matrix != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, matrix, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, matrix[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(target_xyxy)
        if n:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = target_xyxy[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ matrix.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

            # filter candidates
            i = self.box_candidates(target_xyxy.T * s, new.T, area_thr=0.10)
            target_xyxy = new[i]

        return img, target_xyxy

    @staticmethod
    def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


if __name__ == "__main__":
    from utils.bbox import *
    from utils.helper_cv import *

    test_image_file = r"F:\PASCALVOC\VOC2012\JPEGImages\2007_000733.jpg"
    test_image = np.asarray(cv2.imread(test_image_file, flags=cv2.IMREAD_COLOR))  # [H,W,C]
    test_bbox = np.asarray([[48, 25, 273, 383], [103, 201, 448, 435]])  # mode=x1y1x2y2

    enhanceImage, __bbox = RandomPerspective()(test_image, test_bbox)  # [C,H,W]
    print(__bbox)
    show_bbox(enhanceImage, __bbox)
