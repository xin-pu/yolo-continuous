import os.path
import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.cuda import amp

from tqdm import tqdm

from dataset.data_loader import get_dataloader
from learningrate_scheduler import *
from losses.yolo_loss import YOLOLoss
from nets.yolo import Model
from optimizer import get_optimizer
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


def train(train_cfg_file):
    train_cfg = cvt_cfg(train_cfg_file)

    # Step 1 Create Model
    print("Step 1 Create Model")
    model_cfg = cvt_cfg(check_file(train_cfg['model_cfg']))
    model_cfg["nc"] = label_num = train_cfg["num_labels"]
    device = select_device(device='0')
    net = Model(model_cfg).to(device)
    # Todo Resume

    # Step 2 Create Optimizer
    optimizer = get_optimizer(net, train_cfg)  # 以LrI作为初始学习率
    learning_rate_scheduler = get_lr_scheduler(optimizer, 1, train_cfg["lrF"], LearningSchedule.CosineDecay)
    scaler = amp.GradScaler(enabled=True)
    # Todo Resume

    # Step 3 DataLoader
    train_dataloader = get_dataloader(train_cfg, True)
    test_dataloader = get_dataloader(train_cfg, False)
    # Todo

    # Step 4 Loss
    anchors = np.array(model_cfg['anchors']).reshape(-1, 2)
    compute_loss_ota = YOLOLoss(anchors, label_num, (640, 640))

    # Step 5 Train
    epochs = train_cfg["epochs"]
    iterations_each_epoch = len(train_dataloader)
    mean_val_loss_his = []

    model_train = net.train()
    model_train = torch.nn.DataParallel(model_train)

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
                loss = compute_loss_ota(pred, targets, images)  # loss scaled by batch_size

            scaler.scale(loss).backward()

            # Optimize
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Print
            loss_sum += loss.item()
            mean_loss = loss_sum / (i + 1)
            msg = "Epoch:{:05d}\tBatch:{:05d}\tIte:{:05d}\tLoss:{:>.4f}".format(epoch, i, iterations_total, mean_loss)
            pbar.set_description(msg)

        for i, (images, targets) in enumerate(test_dataloader):
            images = images.to(device, non_blocking=True).float()
            targets = targets.to(device)
            with amp.autocast(enabled=True):
                pred = net(images)
                val_loss = compute_loss_ota(pred, targets, images)

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


if __name__ == "__main__":
    _train_cfg_file = check_file(r"cfg/voc_train.yaml")
    train(_train_cfg_file)
