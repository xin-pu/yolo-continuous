from copy import deepcopy
from enum import Enum
from torch import nn
from nets.common import *
from nets.detect import Detect
from nets.iaux_detect import IAuxDetect
from nets.ibin import IBin
from nets.idetect import IDetect


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def parse_model(d, ch, anchors, num_classes):  # model_dict, input_channels(3)
    anchors, nc, gd, gw = anchors, num_classes, d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            # noinspection PyBroadException
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except Exception:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, RobustConv, RobustConv2, dw_conv, GhostConv, RepConv, DownC,
                 SPP, SPPF, SPPCSPC, GhostSPPCSPC, Focus, Stem, GhostStem,
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                 Res, ResCSPA, ResCSPB, ResCSPC,
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC,
                 ResX, ResXCSPA, ResXCSPB, ResXCSPC,
                 RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC,
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC, GhostSPPCSPC,
                     BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                     RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                     ResCSPA, ResCSPB, ResCSPC,
                     RepResCSPA, RepResCSPB, RepResCSPC,
                     ResXCSPA, ResXCSPB, ResXCSPC,
                     RepResXCSPA, RepResXCSPB, RepResXCSPC,
                     GhostCSPA, GhostCSPB, GhostCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Chuncat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Foldcut:
            c2 = ch[f] // 2
        elif m in [Detect, IDetect, IAuxDetect, IBin]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class WeightInitial(Enum):
    NA = 0
    Random = 1


class Model(nn.Module):
    def __init__(self,
                 model_cfg,
                 anchors,
                 num_classes,
                 image_chan=3,
                 weight_initial=WeightInitial.Random):
        super(Model, self).__init__()
        self.traced = False
        self.weight_initial = weight_initial

        self.model, self.save = parse_model(deepcopy(model_cfg),
                                            ch=[image_chan],
                                            anchors=anchors,
                                            num_classes=num_classes)

        # 初始化参数
        self.initial_weights()

    def initial_weights(self):
        if self.weight_initial == WeightInitial.NA:
            return
        elif self.weight_initial == WeightInitial.Random:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.normal_(m.weight, 0, 0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                    m.inplace = True

    def print_info(self):  # print model information
        """
        打印模型信息
        """
        num_parameters = sum(x.numel() for x in self.parameters())
        num_gradients = sum(x.numel() for x in self.parameters() if x.requires_grad)

        print('{0:5s} {1:40s} {2:9s} {3:12s} {4:20s} {5:10s} {6:10s}'.format('layer', 'name', 'gradient', 'parameters',
                                                                             'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(self.named_parameters()):
            name = name.replace('module_list.', '')
            # noinspection PyArgumentList
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
        print('total parameters: {0:7g} total gradients: {1:7g}'.format(num_parameters, num_gradients))

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # noinspection PyTypeChecker
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x


if __name__ == "__main__":
    from utils.helper_io import check_file, cvt_cfg
    from utils.helper_torch import select_device

    _cfg = cvt_cfg(check_file(r"../cfg/net\\yolov7.yaml"))
    _device = select_device(device='0')
    _ch = 3
    _num_classes = 1
    _image_size = 640
    # Create model
    _anchors = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]

    _model = Model(_cfg, _anchors, num_classes=_num_classes, image_chan=_ch).to(_device)
    _model.print_info()
    _model.train()

    _img = torch.rand(1, _ch, _image_size, _image_size).to(_device)
    y1, y2, y3 = _model(_img)

    print("y1:\t{}\r\ny2:\t{}\r\ny3:\t{}".format(tuple(y1.shape), tuple(y2.shape), tuple(y3.shape)))
