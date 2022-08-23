import torch
from torch.cuda import amp

from tqdm import tqdm

from data_loader import get_dataloader
from learningrate_scheduler import *
from losses.compute_loss import ComputeLossOTA
from nets.ModelEMA import ModelEMA
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
    epochs = 2
    iterations_each_epoch = len(train_dataloader)
    iterations_total = 0

    for epoch in range(0, epochs):
        net.train()

        mean_loss = torch.zeros(4, device=device)

        pbar = tqdm(enumerate(train_dataloader), total=iterations_each_epoch)
        optimizer.zero_grad()

        for i, (images, targets) in pbar:
            iterations_total = i + epoch * iterations_each_epoch
            images = images.to(device, non_blocking=True).float() / 255

            with amp.autocast(enabled=True):
                pred = net(images)  # forward
                loss, loss_items = compute_loss_ota(pred, targets.to(device), images)  # loss scaled by batch_size

            scaler.scale(loss).backward()

            # Optimize
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()

            # Print
            mean_loss = (mean_loss * i + loss_items) / (i + 1)
            msg = "{}\t{}\t{}\t{}".format(epoch, iterations_total, i, loss)
            pbar.set_description(msg)

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        learning_rate_scheduler.step()


if __name__ == "__main__":
    _train_cfg_file = check_file(r"cfg/voc_train.yaml")
    train(_train_cfg_file)
