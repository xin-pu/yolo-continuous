import time

import numpy as np
import torch
from torch.cuda import amp
from tqdm import tqdm

from main.data_loader import get_dataloader
from main.learningrate_scheduler import *
from losses.yolo_loss import YOLOLoss
from main.warm_up import warm_up
from nets.yolo import Model, WeightInitial
from main.optimizer import *
from utils.helper_io import check_file, cvt_cfg
from utils.helper_torch import select_device


def print_title(title):
    print("{0}{1:30s}{2}".format("-" * 20, title, "-" * 20))


def train(train_cfg_file):
    print_title("0. 加载计划")
    plan = TrainPlan(train_cfg_file)
    device = select_device(device=plan.device)
    print(plan)

    print_title("1. 构造模型")
    model_cfg = cvt_cfg(check_file(plan.model_cfg))
    net = Model(model_cfg,
                plan.anchors,
                plan.num_labels,
                image_chan=plan.image_chan,
                weight_initial=WeightInitial.Random).to(device)
    net.print_info()
    # net.load_state_dict(torch.load(plan.save_path))

    #  Todo Resume
    resume = False
    if resume:
        model_path = r"resource/yolov7_weights.pth"
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
        print("Resume from {}".format(model_path))

    model_train = torch.nn.DataParallel(net)

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
    mean_loss_his, mean_val_loss_his = [], []

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

            # 优化器更新
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # 打印
            loss_sum += loss.item()
            mean_loss = loss_sum / (i + 1)
            msg = "  Epoch:{:03d}/{:03d}\tBatch:{:05d}\tIte:{:05d}\tLoss:{:>.4f}\tlr:{:>.6f}".format(epoch + 1,
                                                                                                     epochs,
                                                                                                     i,
                                                                                                     iterations_total,
                                                                                                     mean_loss,
                                                                                                     get_lr(optimizer))
            pbar.set_description(msg)
        mean_loss_his.append(mean_loss)

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

        if mean_loss <= min(mean_loss_his):
            torch.save(net.state_dict(), plan.save_path)
            print("Epoch {:05d}  Loss:{:>.4f} ,Val Loss:{:>.4f} save to {}\r\n".format(epoch,
                                                                                       mean_loss,
                                                                                       mean_val_loss,
                                                                                       plan.save_path))

        time.sleep(0.2)


if __name__ == "__main__":
    _train_cfg_file = check_file(r"cfg/raccoon_train.yaml")
    train(_train_cfg_file)
