import math
from enum import Enum

from torch.optim import lr_scheduler
from cfg.train_plan import TrainPlan
from utils.helper_torch import timer


class DecayType(Enum):
    NA = 0
    Linear = 1
    InverseTime = 2
    Exponential = 3
    Natural_Exponential = 4
    Cosine = 5

    @staticmethod
    def get_decay_type(decay):
        if decay == 'Linear':
            return DecayType.Linear
        elif decay == 'InverseTime':
            return DecayType.InverseTime
        elif decay == 'Exponential':
            return DecayType.Exponential
        elif decay == 'Natural_Exponential':
            return DecayType.Natural_Exponential
        elif decay == 'Cosine':
            return DecayType.Cosine
        else:
            return DecayType.NA


def na_decay(y1=0.0, y2=1.0, steps=100):
    """
    线性衰减
    """
    return lambda x: x


def linear_decay(y1=0.0, y2=1.0, steps=100):
    """
    线性衰减
    """
    return lambda x: y2 - (y2 - y1) * (1.0 - x / (steps - 1))


def inverse_time_decay(y1=0.0, y2=1.0, beta=0.1):
    """
    逆时衰减
    """
    return lambda x: y2 - (y2 - y1) / (1 + beta * x)


def exponential_decay(y1=0.0, y2=1.0, beta=0.96):
    """
    指数衰减
    """
    return lambda x: y2 - (y2 - y1) * math.pow(beta, x)


def natural_exponential_decay(y1=0.0, y2=1.0, beta=0.04):
    """
    自然指数衰减
    """
    return lambda x: y2 - (y2 - y1) * math.exp(-beta * x)


def cosine_decay(y1=0.0, y2=1.0, steps=100):
    """
    余弦衰减
    """
    return lambda x: y2 - (y2 - y1) * (1 + math.cos(x * math.pi / steps)) / 2


@timer
def get_lr_scheduler(optimizer, plan: TrainPlan):
    lr_final = plan.learn_final
    epochs = plan.epochs
    decay = plan.decay
    print("使用{}衰减".format(plan.decay))
    ls = DecayType.get_decay_type(decay)
    if ls == DecayType.Linear:
        lf = linear_decay(1, lr_final, epochs)
    elif ls == DecayType.InverseTime:
        lf = inverse_time_decay(1, lr_final, epochs)
    elif ls == DecayType.Exponential:
        lf = exponential_decay(1, lr_final, epochs)
    elif ls == DecayType.Natural_Exponential:
        lf = natural_exponential_decay(1, lr_final, epochs)
    elif ls == DecayType.Cosine:
        lf = cosine_decay(1, lr_final, epochs)
    else:
        lf = na_decay(1, lr_final, epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=-1)
    scheduler.name = decay
    # print(scheduler.get_lr()[0])
    return scheduler




if __name__ == "__main__":
    pass
