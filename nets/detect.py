import torch
from torch import nn


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.classes = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        anchors = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', anchors)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', anchors.clone().view(self.nl, 1, -1, 1, 1, 2))  # keypoint 可以与设备无关

        self.yolo_head_P3 = nn.Conv2d(ch[0], self.na * self.no, 1)
        self.yolo_head_P4 = nn.Conv2d(ch[1], self.na * self.no, 1)
        self.yolo_head_P5 = nn.Conv2d(ch[2], self.na * self.no, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

    def forward(self, x):
        # ---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size, 75, 80, 80)
        # ---------------------------------------------------#
        out2 = self.yolo_head_P3(x[0])
        # ---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size, 75, 40, 40)
        # ---------------------------------------------------#
        out1 = self.yolo_head_P4(x[1])
        # ---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size, 75, 20, 20)
        # ---------------------------------------------------#
        out0 = self.yolo_head_P5(x[2])
        return [out0, out1, out2]

    @staticmethod
    def make_grid(nx=20, ny=20):
        # Fixed  Warning: torch.meshgrid : in an upcoming release, it will be required to
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
