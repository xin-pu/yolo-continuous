import math
from enum import Enum

from torch.optim import lr_scheduler, SGD
from torchvision.models import ResNet

from utils.helper_torch import timer


class LearningSchedule(Enum):
    Linear_Decay = 0
    InverseTime_Decay = 1
    Exponential_Decay = 2
    Natural_Exponential_Decay = 3
    CosineDecay = 4


def linear_decay(y1=0.0, y2=1.0, steps=100):
    """
    线性衰减
    """
    return lambda x: y2 - (y2 - y1) * (1.0 - x / (steps - 1))


def inverse_time_decay(y1=0.0, y2=1.0, beta=0.1):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: y2 - (y2 - y1) / (1 + beta * x)


def exponential_decay(y1=0.0, y2=1.0, beta=0.96):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: y2 - (y2 - y1) * math.pow(beta, x)


def natural_exponential_decay(y1=0.0, y2=1.0, beta=0.04):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: y2 - (y2 - y1) * math.exp(-beta * x)


def cosine_decay(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: y2 - (y2 - y1) * (1 + math.cos(x * math.pi / steps)) / 2


@timer
def get_lr_scheduler(optimizer, lr, epochs, ls=LearningSchedule.CosineDecay):
    if ls == LearningSchedule.CosineDecay:
        lf = cosine_decay(1, lr, epochs)  # cosine 1->hyp['lrf']
    elif ls == LearningSchedule.InverseTime_Decay:
        lf = inverse_time_decay(1, lr, epochs)
    elif ls == LearningSchedule.Exponential_Decay:
        lf = exponential_decay(1, lr, epochs)
    elif ls == LearningSchedule.Natural_Exponential_Decay:
        lf = natural_exponential_decay(1, lr, epochs)
    else:
        lf = linear_decay(1, lr, epochs)  # linear 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler


if __name__ == "__main__":
    model = ResNet()
    d = SGD(model.parameters(), lr=1)
    ll = get_lr_scheduler(model, 0.1, 100)
    for i in range(100):
        ll.step()
        print(d)
