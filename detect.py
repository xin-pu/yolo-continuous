import os

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from cfg.train_plan import TrainPlan
from image_enhance.letter_box import LetterBox
from nets.yolo_net import YoloBody
from utils.helper_io import cvt_cfg, check_file
from utils.helper_torch import select_device


def decode_box(inputs, anchors, anchors_mask, num_labels, image_size=(640, 640)):
    outputs = []
    for i, input in enumerate(inputs):
        # -----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size = 1
        #   batch_size, 3 * (4 + 1 + 80), 20, 20
        #   batch_size, 255, 40, 40
        #   batch_size, 255, 80, 80
        # -----------------------------------------------#
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        # -----------------------------------------------#
        #   输入为640x640时
        #   stride_h = stride_w = 32、16、8
        # -----------------------------------------------#
        stride_h = image_size[0] / input_height
        stride_w = image_size[0] / input_width
        # -------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        # -------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                          anchors[anchors_mask[i]]]

        # -----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 3, 20, 20, 85
        #   batch_size, 3, 40, 40, 85
        #   batch_size, 3, 80, 80, 85
        # -----------------------------------------------#
        prediction = input.view(batch_size, len(anchors_mask[i]),
                                num_labels + 5, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # -----------------------------------------------#
        #   先验框的中心位置的调整参数
        # -----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # -----------------------------------------------#
        #   先验框的宽高调整参数
        # -----------------------------------------------#
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        # -----------------------------------------------#
        #   获得置信度，是否有物体
        # -----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 4])
        # -----------------------------------------------#
        #   种类置信度
        # -----------------------------------------------#
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # ----------------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        #   batch_size,3,20,20
        # ----------------------------------------------------------#
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * len(anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * len(anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

        # ----------------------------------------------------------#
        #   按照网格格式生成先验框的宽高
        #   batch_size,3,20,20
        # ----------------------------------------------------------#
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        # ----------------------------------------------------------#
        #   利用预测结果对先验框进行调整
        #   首先调整先验框的中心，从先验框中心向右下角偏移
        #   再调整先验框的宽高。
        #   x 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
        #   y 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
        #   w 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
        #   h 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
        # ----------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
        pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
        pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
        pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

        # ----------------------------------------------------------#
        #   将输出结果归一化成小数的形式
        # ----------------------------------------------------------#
        _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, num_labels)), -1)
        outputs.append(output.data)
    return outputs


def non_max_suppression(prediction,
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
            detections_class = detections[detections[:, -1] == c]

            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres
            )
            max_detections = detections_class[keep]

            # Add max detections to outputs
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

    return output


if __name__ == "__main__":
    _train_cfg_file = check_file(r"cfg/raccoon_train.yaml")
    plan = TrainPlan(_train_cfg_file)
    device = select_device(device='0')

    model_cfg = cvt_cfg(check_file(plan.model_cfg))

    path = os.path.join(plan.save_dir, "{}.pt".format(plan.save_name))
    ckpt = torch.load(path, map_location=device)
    net = YoloBody(plan.anchors_mask, 80, 'l').cuda()
    net.load_state_dict(torch.load(r"D:\Download\yolov7_weights.pth"))

    image_file = r"E:\OneDrive - II-VI Incorporated\Pictures\Saved Pictures\voc\004545.jpg"
    image = cv2.imread(image_file)
    image = LetterBox()(np.array(image), [[0, 0, 0, 0]])[0]

    with torch.no_grad():
        image = image.astype(np.float32) / 255
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
        outputs = net(image)
        anchors = np.asarray(plan.anchors).reshape(-1, 2)
        anchors_mask = plan.anchors_mask
        outputs = decode_box(outputs, anchors, anchors_mask, 80, image_size=(640, 640))
        all_outputs = torch.cat(outputs, 1)
        results = non_max_suppression(all_outputs, 80, (640, 640), (640, 640),
                                      image,
                                      conf_thres=0.5,
                                      nms_thres=0.5)

        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]
