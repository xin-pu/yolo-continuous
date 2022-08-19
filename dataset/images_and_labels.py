import math
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch.distributed
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

from dataset.infinite_dataLoader import InfiniteDataLoader


class ImagesAndLabels(Dataset):

    def __init__(self,
                 data_cfg_path,
                 enhance_cfg_path,
                 train=True):
        data_cfg = self.get_dataset_cfg(data_cfg_path)
        enhance_cfg = self.get_dataset_cfg(enhance_cfg_path)

        image_index_file = data_cfg["train"] if train else data_cfg["val"]

        self.annot_encode_folder = data_cfg["annot_encode_folder"]
        self.image_files = pd.read_csv(image_index_file, header=None).iloc[:, 0].values
        self.annot_files = self.get_annot_file(self.image_files)
        self.len = len(self.annot_files)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        import random
        s = random.randint(1, 4)
        return torch.ones(3), 4

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            info += "{}:\t{}\r\n".format(key, value)
        return info

    def get_annot_file(self, image_files):
        ann = []
        for f in image_files:
            file_name, extension = os.path.splitext(os.path.basename(f))
            ann.append(os.path.join(self.annot_encode_folder, "{}.txt".format(file_name)))
        return ann

    @staticmethod
    def get_dataset_cfg(cfg_file):
        with open(cfg_file, 'r') as file:
            cfg = yaml.safe_load(file)
            return cfg


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


if __name__ == "__main__":
    _data_cfg = "../cfg/voc_train.yaml"
    _enhance_cfg = "../cfg/enhance/enhance.yaml"
    rank = 1

    dataset = ImagesAndLabels(_data_cfg, _enhance_cfg)
    dataloader = InfiniteDataLoader(dataset, batch_size=32, shuffle=True)
    i = 1
    for images, targets in dataloader:
        i += 1
        print(i)
