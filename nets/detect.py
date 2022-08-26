import torch
from torch import nn


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        anchors = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', anchors)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', anchors.clone().view(self.nl, 1, -1, 1, 1, 2))  # keypoint 可以与设备无关
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
        x.reverse()
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def make_grid(nx=20, ny=20):
        # Fixed  Warning: torch.meshgrid : in an upcoming release, it will be required to
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
