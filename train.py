import os
import time

import numpy as np
import torch
from torch.cuda import amp
from copy import deepcopy
from tqdm import tqdm
from dataset.data_loader import get_dataloader
from utils.learningrate_scheduler import *
from losses.yolo_loss import YOLOLoss
from nets.yolo import Model, WeightInitial
from utils.optimizer import *
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
    print("{0}{1:40s}{2}".format("-" * 20, title, "-" * 20))


def train(train_cfg_file):
    device = select_device(device='0')

    train_cfg = cvt_cfg(train_cfg_file)
    num_classes, anchors = train_cfg["num_labels"], train_cfg['anchors'],
    image_size, image_chan = train_cfg["image_size"], train_cfg['image_chan']
    model_cfg_file = train_cfg['model_cfg']

    print_title("1. 构造模型")
    model_cfg = cvt_cfg(check_file(model_cfg_file))
    net = Model(model_cfg, anchors, num_classes, image_chan=image_chan, weight_initial=WeightInitial.Random).to(device)
    # Todo Resume

    print_title("2. 构造优化器")
    optimizer = get_optimizer(net, train_cfg)
    learning_rate_scheduler = get_lr_scheduler(optimizer, train_cfg["lrF"], train_cfg["epochs"],
                                               LearningSchedule.CosineDecay)
    scaler = amp.GradScaler(enabled=True)

    print_title("3. 构造损失函数")
    anchors = np.array(anchors).reshape(-1, 2)
    yolo_loss = YOLOLoss(anchors, num_classes, (image_size, image_size))

    # Step 3 DataLoader
    print_title("4. 构造数据集")
    train_dataloader = get_dataloader(train_cfg, True)
    test_dataloader = get_dataloader(train_cfg, False)

    # Step 5 Train
    epochs = train_cfg["epochs"]
    iterations_each_epoch = len(train_dataloader)
    mean_val_loss_his = []

    model_train = torch.nn.DataParallel(net)

    for epoch in range(0, epochs):

        pbar = tqdm(enumerate(train_dataloader), total=iterations_each_epoch, ncols=100, colour='#FFFFFF')
        optimizer.zero_grad()
        loss_sum, mean_loss = 0, 0
        val_loss_sum, mean_val_loss = 0, 0

        for i, (images, targets) in pbar:
            iterations_total = i + epoch * iterations_each_epoch
            images = images.to(device).float()
            targets = targets.to(device)
            with amp.autocast(enabled=True):
                pred = model_train(images)  # forward
                loss = yolo_loss(pred, targets, images)  # loss scaled by batch_size

            scaler.scale(loss).backward()

            # Optimize
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Print
            loss_sum += loss.item()
            mean_loss = loss_sum / (i + 1)
            msg = "Epoch:{:05d}\tBatch:{:05d}\tIte:{:05d}\tLoss:{:>.4f}\tlr:{:>.5f}".format(epoch, i, iterations_total,
                                                                                            mean_loss,
                                                                                            get_lr(optimizer))
            pbar.set_description(msg)

        for i, (images, targets) in enumerate(test_dataloader):
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
            path = os.path.join(train_cfg['save_dir'], "best.pt")
            ckpt = {'epoch': epoch,
                    'model': deepcopy(net),
                    'optimizer': optimizer.state_dict()}
            torch.save(ckpt, path)
            print("Epoch {:05d} Val Loss:{:>.4f} save to {}\r\n".format(epoch, mean_val_loss, path))
        time.sleep(0.2)

        learning_rate_scheduler.step()


if __name__ == "__main__":
    _train_cfg_file = check_file(r"cfg/raccoon_train.yaml")
    train(_train_cfg_file)
