from torch import optim, nn
from torch.nn import Module

from cfg.train_plan import TrainPlan
from utils.helper_torch import timer


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


@timer
def get_optimizer(model: Module,
                  train_plan: TrainPlan):
    """
    根据模型和配置文件生成优化器，并将模型权重和偏置传给优化器，附加衰减
    :type train_plan: object
    :param model:
    :return:
    """
    adam = train_plan.adam
    learn_initial = train_plan.learn_initial
    momentum = train_plan.momentum
    weight_decay = train_plan.weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    if adam:
        optimizer = optim.NAdam(pg0, lr=learn_initial, betas=(momentum, 0.999))
    else:
        optimizer = optim.SGD(pg0, lr=learn_initial, momentum=momentum, nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

    del pg0, pg1, pg2

    return optimizer
