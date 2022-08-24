import torch.nn as nn
import torch.nn.functional as f
from losses.components.focal_loss import FocalLoss
from utils.bbox import *


# https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
def smooth_bce(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class ComputeLossOTA:
    def __init__(self, model, train_cfg, auto_balance=False):
        super(ComputeLossOTA, self).__init__()
        self.device = device = next(model.parameters()).device  # Todo get model device
        self.train_cfg = train_cfg = train_cfg
        self.auto_balance = auto_balance

        # Define criteria
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_cfg['cls_pw']], device=device))
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_cfg['obj_pw']], device=device))

        # Class label smoothing
        # https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.bce_positive, self.bce_negative = smooth_bce(eps=train_cfg.get('label_smoothing', 0.0))

        # Focal loss
        fl_gamma = train_cfg['fl_gamma']
        if fl_gamma > 0:
            self.bce_cls = FocalLoss(self.bce_cls, fl_gamma)
            self.bce_obj = FocalLoss(self.bce_obj, fl_gamma)

        detect = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module

        self.gr = train_cfg['iou_loss_ratio']
        self.na = detect.na
        self.nc = detect.nc
        self.nl = detect.nl
        self.anchors = detect.anchors
        self.stride = detect.stride

        self.balance = {3: [4.0, 1.0, 0.4]}.get(detect.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(detect.stride).index(16) if auto_balance else 0  # stride 16 index

    def __call__(self, preds, targets, images):  # predictions, targets, model
        device = self.device

        # 创建最后的格类损失标量
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)

        # 分配标签
        bs, as_, gjs, gis, targets, anchors = self.build_targets(preds, targets, images)

        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in preds]

        # Losses
        for i, pi in enumerate(preds):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack((torch.asarray(gi), torch.asarray(gj)), dim=1)
                pred_xy = ps[:, :2].sigmoid() * 2. - 0.5
                pred_wh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pred_box = torch.cat((pred_xy, pred_wh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pred_box.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                loss_box = loss_box + (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.bce_negative, device=device)  # targets
                    t[range(n), selected_tcls] = self.bce_positive
                    loss_cls = loss_cls + self.bce_cls(ps[:, 5:], t)  # BCE

            obj_i = self.bce_obj(pi[..., 4], tobj)
            loss_obj = loss_obj + obj_i * self.balance[i]  # obj loss

        loss_box *= self.train_cfg['box']
        loss_obj *= self.train_cfg['obj']
        loss_cls *= self.train_cfg['cls']
        bs = tobj.shape[0]  # batch size

        return (loss_box + loss_obj + loss_cls) * bs

    def build_targets(self, preds, targets, images):

        indices, anchors = self.find_3_positive(preds, targets)

        matching_bs = [[] for _ in preds]
        matching_as = [[] for _ in preds]
        matching_gjs = [[] for _ in preds]
        matching_gis = [[] for _ in preds]
        matching_targets = [[] for _ in preds]
        matching_anchors = [[] for _ in preds]

        nl = len(preds)

        for batch_idx in range(preds[0].shape[0]):

            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            xywh_targets = this_target[:, 2:6] * images[batch_idx].shape[1]
            xyxy_targets = cvt_bbox(xywh_targets, CvtFlag.CVT_XYWH_XYXY)

            xyxys_pred = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, pi in enumerate(preds):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anchors[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)

                fg_pred = pi[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i]  # / 8.
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anchors[i][idx] * self.stride[i]  # / 8.
                xywh_pred = torch.cat([pxy, pwh], dim=-1)
                xyxy_pred = cvt_bbox(xywh_pred, CvtFlag.CVT_XYWH_XYXY)
                xyxys_pred.append(xyxy_pred)

            xyxys_pred = torch.cat(xyxys_pred, dim=0)
            if xyxys_pred.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            pair_wise_iou = box_iou(xyxy_targets, xyxys_pred)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                f.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, xyxys_pred.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                    p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = f.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

            cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchors[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchors[i] = torch.cat(matching_anchors[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchors[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchors

    def find_3_positive(self, pred, targets):

        device = self.device
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        num_anchors, num_targets = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=device).long()  # normalized to grid space gain
        ai = torch.arange(num_anchors, device=device).float().view(num_anchors, 1).repeat(1, num_targets)
        targets = torch.cat((targets.repeat(num_anchors, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(pred[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if num_targets:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.train_cfg['anchor_t']  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            anchor_indices = t[:, 6].long()  # anchor indices
            indices.append((b, anchor_indices, gj.clamp_(0, gain[3] - 1),
                            gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[anchor_indices])  # anchors

        return indices, anch


if __name__ == "__main__":
    from utils.helper_io import check_file, cvt_cfg
    from utils.helper_torch import select_device
    from nets.yolo import Model

    _train_cfg_file = check_file(r"../cfg/voc_train.yaml")
    _train_cfg = cvt_cfg(_train_cfg_file)

    # Step 1 Create Model
    print("Step 1 Create Model")
    model_cfg = cvt_cfg(check_file(_train_cfg['model_cfg']))
    _device = select_device(device='0')
    net = Model(model_cfg).to(_device)
    loss = ComputeLossOTA(net, _train_cfg)
