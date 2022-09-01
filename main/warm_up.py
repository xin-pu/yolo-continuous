"""
Author: Xin.PU
Email: Pu.Xin@outlook.com
Time: 2022/9/1 14:21
"""
import numpy as np
from torch.optim import Optimizer

from cfg.train_plan import TrainPlan


def warm_up(optimizer: Optimizer, plan: TrainPlan, iteration_curr, warm_up_iterations):
    range = [0, warm_up_iterations]
    for j, x in enumerate(optimizer.param_groups):
        if j == 2:  # Bias
            x['lr'] = np.interp(iteration_curr, range, [plan.warmup_bias_lr, x['initial_lr']])
        else:  # Weights
            x['lr'] = np.interp(iteration_curr, range, [0.0, x['initial_lr']])
        if 'momentum' in x:
            x['momentum'] = np.interp(iteration_curr, range, [plan.warmup_momentum, plan.momentum])
    pass
