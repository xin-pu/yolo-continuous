from torch.optim import lr_scheduler

from learningrate_scheduler import *
from nets.yolo import Model
from optimizer import get_optimizer
from utils.helper_io import check_file, cvt_cfg
from utils.helper_torch import select_device


def train(train_cfg_file):
    train_cfg = cvt_cfg(train_cfg_file)
    model_cfg = cvt_cfg(check_file(train_cfg['model_cfg']))
    enhance_cfg = cvt_cfg(check_file(train_cfg['enhance_cfg']))

    # Step 1 Create Model
    print("Step 1 Create Model")
    device = select_device(device='0')
    net = Model(model_cfg).to(device)

    # Step 2 Create Optimizer
    optimizer = get_optimizer(net, train_cfg)  # 以LrI作为树池化学习率
    learning_rate_scheduler = get_lr_scheduler(optimizer, 1, train_cfg["lrF"], LearningSchedule.CosineDecay)


if __name__ == "__main__":
    _train_cfg_file = check_file(r"cfg/voc_train.yaml")
    train(_train_cfg_file)
