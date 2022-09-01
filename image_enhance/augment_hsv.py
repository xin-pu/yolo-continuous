from torch.nn import Module
from utils.helper_cv import *


class RandomHSV(Module):
    def __init__(self,
                 hgain=0.015,
                 sgain=0.7,
                 vgain=0.4):
        super(RandomHSV, self).__init__()
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, img: ndarray, target_xyxy):
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return img, target_xyxy


if __name__ == "__main__":
    from utils.bbox import *
    from utils.helper_cv import *

    test_image_file = r"F:\PASCALVOC\VOC2012\JPEGImages\2007_000733.jpg"

    test_image = np.asarray(cv2.imread(test_image_file, flags=cv2.IMREAD_COLOR),dtype=np.float32)  # [H,W,C]
    test_bbox = np.asarray([[48, 25, 273, 383], [103, 201, 448, 435]])  # mode=xyxy

    enhanceImage, __bbox = RandomHSV()(test_image, test_bbox)  # [H,W,C]
    print(__bbox)
    show_bbox(enhanceImage, __bbox)
