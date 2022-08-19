import math
import os
from contextlib import contextmanager

import cv2
import numpy as np
import pandas as pd
import torch.distributed
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

from dataset.infinite_dataLoader import InfiniteDataLoader
from image_enhance.enhance_package import EnhancePackage


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

        self.enhance = EnhancePackage(640, enhance_cfg)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """

        :param index:
        :return:
        img: [C,H,W]
        tar: (Label, X1,Y1,X2,Y2)
        """
        image_file = self.image_files[index]
        target_file = self.annot_files[index]

        img = cv2.imread(image_file)
        tar = np.asarray([[1.0, 1.0, 2.0, 2.0]])

        img, tar = self.enhance(img, tar)

        return torch.from_numpy(img), torch.from_numpy(tar)

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


if __name__ == "__main__":
    _data_cfg = "../cfg/voc_train.yaml"
    _enhance_cfg = "../cfg/enhance/enhance.yaml"
    rank = 1

    dataset = ImagesAndLabels(_data_cfg, _enhance_cfg)
    dataloader = InfiniteDataLoader(dataset, batch_size=32, shuffle=True)
    i = 1
    for images, targets in dataloader:
        i += 1
        break
