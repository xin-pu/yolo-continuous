import os.path
import time
from copy import deepcopy

import torch
from torch.cuda import amp

from tqdm import tqdm

from data_loader import get_dataloader
from learningrate_scheduler import *
from losses.compute_loss import ComputeLossOTA
from nets.yolo import Model
from optimizer import get_optimizer
from utils.helper_io import check_file, cvt_cfg
from utils.helper_torch import select_device


def train(train_cfg_file):
    train_cfg = cvt_cfg(train_cfg_file)

    # Step 1 Create Model
    print("Step 1 Create Model")
    model_cfg = cvt_cfg(check_file(train_cfg['model_cfg']))
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
    compute_loss_ota = ComputeLossOTA(net, train_cfg)

    # Step 5 Train
    epochs = train_cfg["epochs"]
    iterations_each_epoch = len(train_dataloader)
    iterations_total = 0
    mean_val_loss_his = []

    for epoch in range(0, epochs):
        net.train()

        pbar = tqdm(enumerate(train_dataloader), total=iterations_each_epoch, ncols=100, colour='#FFFFFF')
        optimizer.zero_grad()
        loss_sum, mean_loss = 0, 0
        val_loss_sum, mean_val_loss = 0, 0

        for i, (images, targets) in pbar:
            iterations_total = i + epoch * iterations_each_epoch
            images = images.to(device, non_blocking=True).float()
            targets = targets.to(device)
            with amp.autocast(enabled=True):
                pred = net(images)  # forward
                loss = compute_loss_ota(pred, targets, images)  # loss scaled by batch_size

            scaler.scale(loss).backward()

            # Optimize
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()

            # Print
            loss_sum += loss.item() / images.shape[0]
            mean_loss = loss_sum / (i + 1)
            msg = "Epoch:{:05d}\tBatch:{:05d}\tIte:{:05d}\tLoss:{:>.4f}".format(epoch, i, iterations_total, mean_loss)
            pbar.set_description(msg)

        for i, (images, targets) in enumerate(test_dataloader):
            images = images.to(device, non_blocking=True).float()
            targets = targets.to(device)
            with amp.autocast(enabled=True):
                pred = net(images)  # forward
                val_loss = compute_loss_ota(pred, targets, images)

            val_loss_sum += val_loss.item() / images.shape[0]
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

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        learning_rate_scheduler.step()


if __name__ == "__main__":
    _train_cfg_file = check_file(r"cfg/voc_train.yaml")
    train(_train_cfg_file)
