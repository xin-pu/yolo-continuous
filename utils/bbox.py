import math
import numpy as np
import torch
from numpy import ndarray
from enum import Enum
from torch import Tensor
from torchvision.ops import nms


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


def cvt_bbox(bbox: Tensor | ndarray, flag: CvtFlag):
    if not check(flag):
        raise Exception()
    if isinstance(bbox, np.ndarray):
        res_bbox = np.empty_like(bbox)
    else:
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


def box_iou(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_iou(box1, box2, x1y1x2y2=True, giou=False, diou=False, ciou=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if giou or diou or ciou:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if ciou or diou:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if diou:
                return iou - rho2 / c2  # DIoU
            elif ciou:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def non_max_suppression(self,
                        prediction,
                        num_classes,
                        input_shape,
                        image_shape,
                        letterbox_image,
                        conf_thres=0.5,
                        nms_thres=0.4):
    # ----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    # ----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # ----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        # ----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        # ----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        # ----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        # ----------------------------------------------------------#
        #   根据置信度进行预测结果的筛选
        # ----------------------------------------------------------#
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        # -------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        # -------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # ------------------------------------------#
        #   获得预测结果中包含的所有种类
        # ------------------------------------------#
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            # ------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            # ------------------------------------------#
            detections_class = detections[detections[:, -1] == c]

            # ------------------------------------------#
            #   使用官方自带的非极大抑制会速度更快一些！
            #   筛选出一定区域内，属于同一种类得分最大的框
            # ------------------------------------------#
            keep = nms(detections_class[:, :4],
                       detections_class[:, 4] * detections_class[:, 5],
                       nms_thres)
            max_detections = detections_class[keep]

            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output


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
