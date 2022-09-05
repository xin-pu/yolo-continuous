import time

import numpy as np
import torch
from torch.cuda import amp
from copy import deepcopy
from tqdm import tqdm

from main.data_loader import get_dataloader
from main.learningrate_scheduler import *
from losses.yolo_loss import YOLOLoss
from main.warm_up import warm_up
from nets.yolo import Model, WeightInitial
from main.optimizer import *
from nets.yolo_net import YoloBody
from utils.helper_io import check_file, cvt_cfg
from utils.helper_torch import select_device

try:
    from torch._C import cudnn
except ImportError:
    cudnn = None  # type: ignore[assignment]


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


def print_title(title):
    print("{0}{1:30s}{2}".format("-" * 20, title, "-" * 20))


def train(train_cfg_file):
    print_title("0. 加载计划")
    plan = TrainPlan(train_cfg_file)
    device = select_device(device=plan.device)
    print(plan)

    print_title("1. 构造模型")
    # model_cfg = cvt_cfg(check_file(plan.model_cfg))
    # net = Model(model_cfg,
    #             plan.anchors,
    #             plan.num_labels,
    #             image_chan=plan.image_chan,
    #             weight_initial=WeightInitial.Random).to(device)
    # net.print_info()

    net = YoloBody(plan.anchors_mask, plan.num_labels, 'l')
    weights_init(net)
    model_path = r"resource/yolov7_weights.pth"

    print('Load weights {}.'.format(model_path))

    # ------------------------------------------------------#
    #   根据预训练权重的Key和模型的Key进行加载
    # ------------------------------------------------------#
    model_dict = net.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    net.load_state_dict(model_dict)

    model_train = torch.nn.DataParallel(net)
    # Todo Resume

    print_title("2. 构造优化器")
    optimizer = get_optimizer(net, plan)
    learning_rate_scheduler = get_lr_scheduler(optimizer, plan)
    scaler = amp.GradScaler(enabled=True)

    print_title("3. 构造损失函数")
    anchors = np.array(plan.anchors).reshape(-1, 2)
    yolo_loss = YOLOLoss(anchors, plan.num_labels, (plan.image_size, plan.image_size))
    print(yolo_loss)

    print_title("4. 构造数据集")
    train_dataloader = get_dataloader(plan, True)
    test_dataloader = get_dataloader(plan, False)

    print_title("5. 训练")
    epochs = plan.epochs
    iterations_each_epoch = len(train_dataloader)
    iterations_limit = max(plan.warmup_max_iter, iterations_each_epoch * plan.warmup_epochs)
    mean_val_loss_his = []

    for epoch in range(0, epochs):
        pbar = tqdm(enumerate(train_dataloader), total=iterations_each_epoch, ncols=120, colour='#FFFFFF')

        loss_sum, mean_loss = 0, 0
        val_loss_sum, mean_val_loss = 0, 0
        optimizer.zero_grad()

        for i, (images, targets) in pbar:
            iterations_total = i + epoch * iterations_each_epoch
            images = images.to(device).float()
            targets = targets.to(device)

            # 学习率预热
            if plan.warmup and epoch < plan.warmup_epochs and iterations_total < iterations_limit:
                warm_up(optimizer, plan, iterations_total, iterations_limit)

            with amp.autocast(enabled=True):
                pred = model_train(images)
                loss = yolo_loss(pred, targets, images)

            scaler.scale(loss).backward()

            # Optimize
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Print
            loss_sum += loss.item()
            mean_loss = loss_sum / (i + 1)
            msg = "  Epoch:{:03d}/{:03d}\tBatch:{:05d}\tIte:{:05d}\tLoss:{:>.4f}\tlr:{:>.6f}".format(epoch + 1,
                                                                                                     epochs,
                                                                                                     i,
                                                                                                     iterations_total,
                                                                                                     mean_loss,
                                                                                                     get_lr(optimizer))
            pbar.set_description(msg)

        learning_rate_scheduler.step()

        for j, (images, targets) in enumerate(test_dataloader):
            images = images.to(device, non_blocking=True).float()
            targets = targets.to(device)
            with amp.autocast(enabled=True):
                pred = net(images)
                val_loss = yolo_loss(pred, targets, images)

            val_loss_sum += val_loss.item()
        mean_val_loss = val_loss_sum / (len(test_dataloader))
        mean_val_loss_his.append(mean_val_loss)
        pbar.close()

        if mean_val_loss <= min(mean_val_loss_his):
            torch.save(net.state_dict(), plan.save_path)
            print("Epoch {:05d} Val Loss:{:>.4f} save to {}\r\n".format(epoch, mean_val_loss, plan.save_path))
        time.sleep(0.2)


if __name__ == "__main__":
    _train_cfg_file = check_file(r"cfg/raccoon_train.yaml")
    train(_train_cfg_file)
