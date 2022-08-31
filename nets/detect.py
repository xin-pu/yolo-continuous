from torch import nn


class Detect(nn.Module):

    def __init__(self, num_classes=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.len_output = num_classes + 5
        self.num_layers = len(anchors)
        self.num_anchors_each_layer = len(anchors[0]) // 2

        self.yolo_head_P3 = nn.Conv2d(ch[0], self.num_anchors_each_layer * self.len_output, 1)
        self.yolo_head_P4 = nn.Conv2d(ch[1], self.num_anchors_each_layer * self.len_output, 1)
        self.yolo_head_P5 = nn.Conv2d(ch[2], self.num_anchors_each_layer * self.len_output, 1)

        #  KeyPoint  其他参数初始化方法 https://arxiv.org/abs/1708.02002 section 3.3
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

    def forward(self, x):
        #   第一个特征层 y1=(batch_size, 75, 20, 20)
        out0 = self.yolo_head_P5(x[2])
        #   第二个特征层  y2=(batch_size, 75, 40, 40)
        out1 = self.yolo_head_P4(x[1])
        #   第三个特征层 y3=(batch_size, 75, 80, 80)
        out2 = self.yolo_head_P3(x[0])

        if not self.training:
            pass

        return [out0, out1, out2]
