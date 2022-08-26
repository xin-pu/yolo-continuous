from torch import optim, nn
from torch.nn import Module

from utils.helper_torch import timer


@timer
def get_optimizer(model: Module,
                  cfg):
    """
    根据模型和配置文件生成优化器，并将模型权重和偏置传给优化器，附加衰减
    :param model:
    :param cfg:
    :return:
    """
    total_batch_size = 16
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    cfg['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if cfg['adam']:
        optimizer = optim.Adam(pg0, lr=cfg['lrI'], betas=(cfg['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=cfg['lrI'], momentum=cfg['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': cfg['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

    del pg0, pg1, pg2

    return optimizer
