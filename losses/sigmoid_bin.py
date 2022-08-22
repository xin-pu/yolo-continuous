import torch
from torch.nn import Module
from torch.nn import BCEWithLogitsLoss, MSELoss


class SigmoidBin(Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self,
                 bin_count=10,
                 min=0.0,
                 max=1.0,
                 reg_scale=2.0,
                 use_loss_regression=True,
                 use_fw_regression=True,
                 bce_weight=1.0,
                 smooth_eps=0.0):
        super(SigmoidBin, self).__init__()

        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min = min
        self.max = max
        self.scale = float(max - min)
        self.shift = self.scale / 2.0

        self.use_loss_regression = use_loss_regression
        self.use_fw_regression = use_fw_regression
        self.reg_scale = reg_scale
        self.BCE_weight = bce_weight

        start = min + (self.scale / 2.0) / self.bin_count
        end = max - (self.scale / 2.0) / self.bin_count
        self.step = step = self.scale / self.bin_count

        bins = torch.range(start, end + 0.0001, step).float()
        self.register_buffer('bins', bins)

        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps

        self.BCE_bins = BCEWithLogitsLoss(pos_weight=torch.Tensor([bce_weight]))
        self.MSE = MSELoss()

    def get_length(self):
        return self.length

    def forward(self, pred):

        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale / 2.0) * self.step
        pred_bin = pred[..., 1:(1 + self.bin_count)]

        _, bin_idx = torch.max(pred_bin, dim=-1)
        bin_bias = self.bins[bin_idx]

        if self.use_fw_regression:
            result = pred_reg + bin_bias
        else:
            result = bin_bias
        result = result.clamp(min=self.min, max=self.max)

        return result

    def training_loss(self, pred, target):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (
            pred.shape[-1], self.length)
        assert pred.shape[0] == target.shape[0], 'pred.shape=%d is not equal to the target.shape=%d' % (
            pred.shape[0], target.shape[0])
        device = pred.device

        pred_reg = (pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale / 2.0) * self.step
        pred_bin = pred[..., 1:(1 + self.bin_count)]

        diff_bin_target = torch.abs(target[..., None] - self.bins)
        _, bin_idx = torch.min(diff_bin_target, dim=-1)

        bin_bias = self.bins[bin_idx]
        bin_bias.requires_grad = False
        result = pred_reg + bin_bias

        target_bins = torch.full_like(pred_bin, self.cn, device=device)  # targets
        n = pred.shape[0]
        target_bins[range(n), bin_idx] = self.cp

        loss_bin = self.BCE_bins(pred_bin, target_bins)  # BCE

        if self.use_loss_regression:
            loss_regression = self.MSE(result, target)  # MSE
            loss = loss_bin + loss_regression
        else:
            loss = loss_bin

        out_result = result.clamp(min=self.min, max=self.max)

        return loss, out_result
