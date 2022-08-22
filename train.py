from torch.optim import lr_scheduler

from nets.yolo import Model
from optimizer import get_optimizer
from utils.helper_io import check_file, cvt_cfg
from utils.helper_torch import select_device


def train(train_cfg, model_cfg, enhance_cfg):
    model_cfg = cvt_cfg(_model_cfg_file)
    train_cfg = cvt_cfg(_train_cfg_file)

    # Step 1 Create Model
    _device = select_device(device='0')
    _model = Model(model_cfg).to(_device)

    # Step 2 Create Optimizer
    optimizer = get_optimizer(_model, train_cfg)


if __name__ == "__main__":
    _model_cfg_file = check_file(r"cfg/net/yolov7.yaml")  #
    _train_cfg_file = check_file(r"cfg/voc_train.yaml")
    _enhance_cfg_file = check_file(r"cfg/enhance/enhance.yaml")

    train(_train_cfg_file, _model_cfg_file, _enhance_cfg_file)
