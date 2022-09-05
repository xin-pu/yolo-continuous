import cv2
import numpy as np
import torch

from PIL import Image
from PIL.Image import Resampling

from torchvision.ops import nms

from cfg.train_plan import TrainPlan
from nets.yolo import Model, WeightInitial
from nets.yolo_net import YoloBody
from utils.bbox import BBoxType
from utils.helper_cv import show_bbox
from utils.helper_io import check_file, cvt_cfg
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

            keep = nms(detections_class[:, :4], detections_class[:, 4] * detections_class[:, 5], nms_thres)
            max_detections = detections_class[keep]

            # Add max detections to outputs
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4] = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

    return output


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


# ---------------------------------------------------#
#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)


        image = image.resize((nw, nh), Resampling.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Resampling.BICUBIC)
    return new_image


def prepare_model(plan: TrainPlan):
    # cfg = cvt_cfg(plan.model_cfg)
    # net = Model(cfg,
    #             plan.anchors,
    #             plan.num_labels,
    #             image_chan=plan.image_chan,
    #             weight_initial=WeightInitial.Random)
    net = YoloBody(plan.anchors_mask, plan.num_labels, 'l')
    # net.load_state_dict(torch.load(r"E:\ObjectDetect\yolov7_pytorch\logs\best_epoch_weights.pth"))
    net.load_state_dict(torch.load(plan.save_path))
    #   Keypoint 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，pytorch框架会自动把BN和Dropout固定住，不会取平均
    #    而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果。
    # https://blog.csdn.net/wuqingshan2010/article/details/106013660
    net = net.eval()
    return net


def prepare_test_image(image_path):
    image = Image.open(image_path)
    image_data = resize_image(image, (_plan.image_size, _plan.image_size), True)
    image_data = np.expand_dims(np.transpose((np.array(image_data, dtype='float32') / 255.), (2, 0, 1)), 0)
    return image_data, image


if __name__ == "__main__":
    _train_cfg_file = check_file(r"cfg/raccoon_train.yaml")
    _test_img = r"E:\OneDrive - II-VI Incorporated\Pictures\Saved Pictures\raccoon\Racccon (2).jfif"

    _plan = TrainPlan(_train_cfg_file)
    _device = select_device(device=_plan.device)

    _input_shape = (_plan.image_size, _plan.image_size)
    _num_labels = _plan.num_labels
    _anchors = np.asarray(_plan.anchors).reshape(-1, 2)
    _anchors_mask = _plan.anchors_mask

    _image_data, _image = prepare_test_image(_test_img)

    with torch.no_grad():
        _net = prepare_model(_plan).to(_device)
        images = torch.from_numpy(_image_data).to(_device)
        pred = _net(images)

        outputs = decode_box(pred, _anchors, _anchors_mask, _num_labels, image_size=_input_shape)
        all_outputs = torch.cat(outputs, 1)
        results = non_max_suppression(all_outputs, _num_labels, _input_shape, np.array(np.shape(_image)[0:2]),
                                      True,
                                      conf_thres=0.5,
                                      nms_thres=0.3)
        print(results)
        if results[0] is not None:
            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = (results[0][:, 4] * results[0][:, 5])
            top_boxes_yxyx = results[0][:, :4]
            top_boxes_xyxy = np.empty_like(top_boxes_yxyx)
            top_boxes_xyxy[..., 0] = top_boxes_yxyx[..., 1]
            top_boxes_xyxy[..., 1] = top_boxes_yxyx[..., 0]
            top_boxes_xyxy[..., 2] = top_boxes_yxyx[..., 3]
            top_boxes_xyxy[..., 3] = top_boxes_yxyx[..., 2]

            show_bbox(cv2.imread(_test_img), top_boxes_xyxy, bbox_mode=BBoxType.XYXY)
