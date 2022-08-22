from data_loader import get_dataloader
from learningrate_scheduler import *
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
    # Todo Resume

    # Step 3 DataLoader
    train_dataloader = get_dataloader(train_cfg, True)
    test_dataloader = get_dataloader(train_cfg, False)


if __name__ == "__main__":
    _train_cfg_file = check_file(r"cfg/voc_train.yaml")
    train(_train_cfg_file)
