from enum import Enum

import torch
from torch import Tensor


class BBoxType(Enum):
    XYXY = 0
    XXYY = 2
    XYWH = 1


class CvtFlag(Enum):
    CVT_XXYY_XYXY = 0
    CVT_XXYY_XYWH = 1
    CVT_XYXY_XXYY = 2
    CVT_XYXY_XYWH = 3
    CVT_XYWH_XXYY = 4
    CVT_XYWH_XYXY = 5


def check(flag: CvtFlag):
    return True if flag.value in range(6) else False


def cvt_bbox(bbox, flag: CvtFlag):
    if not check(flag):
        raise Exception()
    res_bbox = bbox.clone()
    if flag == CvtFlag.CVT_XXYY_XYXY:
        res_bbox[:, 1] = bbox[:, 2]
        res_bbox[:, 2] = bbox[:, 1]
    if flag == CvtFlag.CVT_XXYY_XYWH:
        res_bbox[:, 0] = (bbox[:, 0] + bbox[:, 1]) / 2  # x center
        res_bbox[:, 1] = (bbox[:, 2] + bbox[:, 3]) / 2  # y center
        res_bbox[:, 2] = bbox[:, 1] - bbox[:, 0]  # width
        res_bbox[:, 3] = bbox[:, 3] - bbox[:, 2]  # height
    if flag == CvtFlag.CVT_XYXY_XXYY:
        res_bbox[:, 1] = bbox[:, 2]
        res_bbox[:, 2] = bbox[:, 1]
    if flag == CvtFlag.CVT_XYXY_XYWH:
        res_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2  # x center
        res_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2  # y center
        res_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  # width
        res_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  # height
    if flag == CvtFlag.CVT_XYWH_XXYY:
        res_bbox[:, 0] = bbox[:, 0] - bbox[:, 2] / 2  # top left x
        res_bbox[:, 1] = bbox[:, 0] + bbox[:, 2] / 2  # bottom right x
        res_bbox[:, 2] = bbox[:, 1] - bbox[:, 3] / 2  # top left y
        res_bbox[:, 3] = bbox[:, 1] + bbox[:, 3] / 2  # bottom right y
    if flag == CvtFlag.CVT_XYWH_XYXY:
        res_bbox[:, 0] = bbox[:, 0] - bbox[:, 2] / 2  # top left x
        res_bbox[:, 1] = bbox[:, 1] - bbox[:, 3] / 2  # top left y
        res_bbox[:, 2] = bbox[:, 0] + bbox[:, 2] / 2  # bottom right x
        res_bbox[:, 3] = bbox[:, 1] + bbox[:, 3] / 2  # bottom right y

    return res_bbox


if __name__ == "__main__":
    _xxyy = torch.asarray([[1, 2, 3, 4]]).float()

    print("XXYY:{}".format(_xxyy))
    _xyxy = cvt_bbox(_xxyy, CvtFlag.CVT_XXYY_XYXY)
    _xywh = cvt_bbox(_xxyy, CvtFlag.CVT_XXYY_XYWH)

    print("XYXY:{}".format(_xyxy))
    print("XYWH:{}".format(_xywh))

    _xyxy = cvt_bbox(_xywh, CvtFlag.CVT_XYWH_XYXY)
    _xxyy = cvt_bbox(_xywh, CvtFlag.CVT_XYWH_XXYY)
    print("XYXY:{}".format(_xyxy))
    print("XXYY:{}".format(_xxyy))

    _xxyy = cvt_bbox(_xyxy, CvtFlag.CVT_XYXY_XXYY)
    _xywh = cvt_bbox(_xyxy, CvtFlag.CVT_XYXY_XYWH)
    print("XYXY:{}".format(_xxyy))
    print("XYWH:{}".format(_xywh))
