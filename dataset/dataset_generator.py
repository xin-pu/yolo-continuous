import os
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *


class ImageDataSet(Dataset):

    def __init__(self,
                 data_cfg_path,
                 enhance_cfg_path):
        """

        :param data_cfg:
        """

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        pass


if __name__ == "__main__":
    pass
