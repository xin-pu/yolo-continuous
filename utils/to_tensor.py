import torch
from torch import Tensor
from torch.nn import Module

cpu = False


def cvt_tensor(tensor: Tensor):
    return tensor.cpu() if (not torch.cuda.is_available() or cpu) else tensor.cuda()


def cvt_module(module: Module):
    return module.cpu() if (not torch.cuda.is_available() or cpu) else module.cuda()


if __name__ == "__main__":
    t = torch.tensor([1, 2]).cuda()
    print(cvt_tensor(t))

    d = torch.tensor((2, 3))
    print(cvt_tensor(d))
