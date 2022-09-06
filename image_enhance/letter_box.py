from random import Random
from utils.helper_cv import *
import numpy as np
import cv2
from numpy import ndarray
from torch.nn import Module


class LetterBox(Module):
    def __init__(self,
                 new_shape=(640, 640),
                 scale_fill_prob=1,
                 color=(114, 114, 114), ):
        """
        当模型输入为正方形时直接将长方形图片resize为正方形会使得图片失真，
        通过填充边界(通常是灰色填充)的方式来保持原始图片的长宽比例
        :param new_shape:目标形状
        :param scale_fill_prob:是否直接拉升
        :param color: 填充颜色
        """
        super(LetterBox, self).__init__()

        self.new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
        self.color = color
        self.scale_fill_prob = scale_fill_prob

    def __call__(self, img: ndarray, target_xyxy):
        new_shape = self.new_shape
        color = self.color

        ran = Random()

        scale_fill = ran.random() < self.scale_fill_prob

        h_original, w_original = img.shape[:2]
        ratio = (self.new_shape[0] / w_original, self.new_shape[1] / h_original)  # 调整后/调整前,width,height
        dw, dh = 0, 0  # w,h
        if scale_fill:  # 任何情况直接拉升
            img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
            ratio = (img.shape[0] / w_original, img.shape[1] / h_original)

        else:
            r = min(ratio)
            ratio = (r, r)
            shape_resize = (int(round(w_original * r)), int(round(h_original * r)))
            dw, dh = new_shape[0] - shape_resize[0], new_shape[1] - shape_resize[1]
            dw /= 2  # divide padding into 2 sides
            dh /= 2

            if new_shape[::-1] != shape_resize:
                img = cv2.resize(img, shape_resize, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        target_xyxy = np.copy(target_xyxy)
        target_xyxy[:, [0, 2]] = target_xyxy[..., [0, 2]] * ratio[0] + dw  # top left x
        target_xyxy[:, [1, 3]] = target_xyxy[..., [1, 3]] * ratio[1] + dh  # top left y

        return img, target_xyxy


if __name__ == "__main__":
    test_image_file = r"F:\PASCALVOC\VOC2007_Val\JPEGImages\001919.jpg"

    test_image = np.asarray(cv2.imread(test_image_file, flags=cv2.IMREAD_COLOR))  # [H,W,C]
    test_bbox = np.asarray([[203, 85, 350, 212]])  # mode=xyxy
    enhanceImage, __bbox = LetterBox(new_shape=(640, 640), scale_fill_prob=0)(test_image, test_bbox)  # [H,W,C]
    print(enhanceImage.shape)
    show_bbox(enhanceImage, __bbox)
