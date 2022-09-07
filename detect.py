import cv2
import numpy as np
import torch

from numpy import ndarray
from torchvision.ops import nms
from cfg.train_plan import TrainPlan
from image_enhance.letter_box import LetterBox
from nets.yolo import Model, WeightInitial
from utils.helper_cv import generate_colors
from utils.helper_io import check_file, cvt_cfg
from utils.helper_torch import select_device
from utils.target_box import TargetBox


def prepare_test_image(image_path, target_size):
    """
    图像预处理
    :param image_path: 图像路径
    :param target_size: 网络输入尺寸
    :return:
    """
    image = cv2.imread(image_path)
    image_data, _ = LetterBox(target_size, scale_fill_prob=0)(image, np.zeros((0, 4)))
    image_data = np.expand_dims(np.transpose((np.array(image_data, dtype='float32') / 255.), (2, 0, 1)), 0)
    return image_data, image


def decode_box(inputs, anchors, anchors_mask, num_labels, image_size=(640, 640)):
    outputs = []
    for i, pred in enumerate(inputs):
        #   输入的input一共有三个，他们的shape分别是   batch_size, 3 * (4 + 1 + num_cls), 20, 20
        batch_size = pred.size(0)
        input_height = pred.size(2)
        input_width = pred.size(3)

        #   输入为640x640时 stride_h = stride_w = 32、16、8
        stride_h = image_size[0] / input_height
        stride_w = image_size[0] / input_width

        #   此时获得的scaled_anchors大小是相对于特征层的
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                          anchors[anchors_mask[i]]]

        prediction = pred.view(batch_size, len(anchors_mask[i]), num_labels + 5, input_height, input_width) \
            .permute(0, 1, 3, 4, 2).contiguous()

        prediction = torch.sigmoid(prediction)

        x, y, w, h = prediction[..., 0], prediction[..., 1], prediction[..., 2], prediction[..., 3]

        #   获得置信度，是否有物体,#   种类置信度
        conf = prediction[..., 4]
        pred_cls = prediction[..., 5:]

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        #   生成网格，先验框中心，网格左上角 batch_size,3,20,20
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * len(anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * len(anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

        #   按照网格格式生成先验框的宽高  batch_size,3,20,20
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        #   利用预测结果对先验框进行调整，首先调整先验框的中心，从先验框中心向右下角偏移，  再调整先验框的宽高。
        #   x 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
        #   y 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
        #   w 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
        #   h 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
        pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
        pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
        pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

        #   将输出结果归一化成小数的形式
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
    #   将预测结果的格式转换成左上角右下角的格式。  prediction  [batch_size, num_anchors, 85]
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        #   对种类预测部分取max。 class_conf  [num_anchors, 1]    种类置信度  class_pred  [num_anchors, 1]    种类
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        #   利用置信度进行第一轮筛选
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        #   根据置信度进行预测结果的筛选
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue

        #   detections  [num_anchors, 7] 7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        #   获得预测结果中包含的所有种类
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


def prepare_model(plan: TrainPlan):
    cfg = cvt_cfg(plan.model_cfg)
    net = Model(cfg,
                plan.anchors,
                plan.num_labels,
                image_chan=plan.image_chan,
                weight_initial=WeightInitial.Random)
    net.load_state_dict(torch.load(plan.save_path))
    #   Keypoint 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，pytorch框架会自动把BN和Dropout固定住，不会取平均
    #    而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果。
    # https://blog.csdn.net/wuqingshan2010/article/details/106013660
    net = net.eval()
    return net


def show_bbox(image: ndarray, target_boxes):
    image = np.ascontiguousarray(image)
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 1
    for target_box in target_boxes:
        tl, br = target_box.get_topleft(), target_box.get_bottomright()
        cv2.rectangle(image, tl, br, target_box.color, 1)

        info = '{} {:.2f}'.format(target_box.label, target_box.score)

        label_size, _ = cv2.getTextSize(info, font, font_size, 1)
        left = target_box.left
        top = target_box.top

        text_tl = np.array((left, top - label_size[1]) if top > label_size[1] else (left, top + 1))
        text_br = text_tl + np.array(label_size)
        text_bl = (text_tl[0], text_br[1])

        cv2.rectangle(image, tuple(text_tl), tuple(text_br), color=target_box.color, thickness=-1)
        cv2.putText(image, info, text_bl, font, font_size, (255, 255, 255))

    cv2.imshow("Predict", image)
    cv2.waitKey()


def predict(cfg_file, image_path, conf_threshold=0.3, nms_threshold=0.3):
    train_cfg_file = check_file(cfg_file)
    plan = TrainPlan(train_cfg_file)

    device = select_device(device=plan.device)

    target_shape = (plan.image_size, plan.image_size)
    num_labels = plan.num_labels
    anchors = np.asarray(plan.anchors).reshape(-1, 2)
    anchors_mask = plan.anchors_mask

    colors = generate_colors(plan.num_labels)

    image_data, original_image = prepare_test_image(image_path, target_shape)
    original_image_shape = np.array(np.shape(original_image)[0:2])

    with torch.no_grad():
        _net = prepare_model(plan).to(device)
    images = torch.from_numpy(image_data).to(device)
    pred = _net(images)

    outputs = decode_box(pred, anchors, anchors_mask, num_labels, image_size=target_shape)
    all_outputs = torch.cat(outputs, 1)
    results = non_max_suppression(all_outputs, num_labels, target_shape, original_image_shape,
                                  True,
                                  conf_thres=conf_threshold,
                                  nms_thres=nms_threshold)

    if results[0] is not None:
        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = (results[0][:, 4] * results[0][:, 5])
        top_boxes_yxyx = results[0][:, :4]
        top_boxes_xyxy = np.empty_like(top_boxes_yxyx)
        top_boxes_xyxy[..., 0] = top_boxes_yxyx[..., 1]
        top_boxes_xyxy[..., 1] = top_boxes_yxyx[..., 0]
        top_boxes_xyxy[..., 2] = top_boxes_yxyx[..., 3]
        top_boxes_xyxy[..., 3] = top_boxes_yxyx[..., 2]

        i = 0
        target_boxes = []
        for label in top_label:
            box = top_boxes_xyxy[i]
            conf = top_conf[i]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            x1 = max(0, np.floor(x1).astype('int32'))
            y1 = max(0, np.floor(y1).astype('int32'))
            x2 = min(original_image.shape[1], np.floor(x2).astype('int32'))
            y2 = min(original_image.shape[0], np.floor(y2).astype('int32'))
            box = [x1, y1, x2, y2]

            label_name = plan.labels[label]
            color = colors[label]
            target_box = TargetBox(box, conf, label_name, color)
            print(target_box)
            target_boxes.append(target_box)
            i = i + 1

        show_bbox(original_image, target_boxes)


if __name__ == "__main__":
    predict(r"cfg/voc_train.yaml",
            r"E:\OneDrive - II-VI Incorporated\Pictures\Saved Pictures\voc\004545.jpg",
            conf_threshold=0.1,
            nms_threshold=0.3)
