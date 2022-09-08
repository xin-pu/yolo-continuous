"""
Author: Xin.PU
Email: Pu.Xin@outlook.com
Time: 2022/9/8 16:55
"""

# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
import torch
from torch import nn
from torch.nn import Flatten, Linear

from nets.backbone import Backbone, Conv
from nets.yolo_net import SPPCSPC


class YoloBody(nn.Module):
    def __init__(self, phi, pretrained=False):
        super(YoloBody, self).__init__()
        # -----------------------------------------------#
        #   定义了不同yolov7版本的参数
        # -----------------------------------------------#
        transition_channels = {'l': 4, 'x': 40}[phi]
        block_channels = 16

        n = {'l': 4, 'x': 6}[phi]

        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 1024
        # ---------------------------------------------------#
        self.backbone = Backbone(transition_channels, block_channels, n, phi, pretrained=pretrained)

        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16)
        self.conv_for_P5 = Conv(transition_channels * 16, transition_channels * 8)

        self.flatten = Flatten()
        self.dense = Linear(5408, 16)

    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)

        P5 = self.sppcspc(feat3)
        P5_conv = self.conv_for_P5(P5)
        f = self.flatten(P5_conv)
        f = self.dense(f)

        return f


if __name__ == "__main__":
    model = YoloBody('l', pretrained=False)
    y = model(torch.ones(size=(1, 3, 416, 416)))
    print(model)
    print(sum(x.numel() for x in model.parameters()))
