from random import Random, sample, shuffle

import cv2
import numpy as np
import torch.distributed
import torch

from torch.utils.data import Dataset
from tqdm import tqdm

from cfg.train_plan import TrainPlan
from dataset.infinite_dataLoader import InfiniteDataLoader
from main.enhance_package import EnhancePackage
from utils.bbox import cvt_bbox, CvtFlag
from utils.helper_io import cvt_cfg


# Keypoint 数据裁剪函数，其实将一个batch下不同数量的Label,Box 以序号做索引拼接成一个张量
def collate_fn(batch):
    img, label = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0)


class YoloDataset(Dataset):

    def __init__(self,
                 train_plan: TrainPlan,
                 enhance_cfg,
                 train=True):

        self.image_shape = (train_plan.image_size, train_plan.image_size)
        with open(train_plan.train_indexes if train else train_plan.val_indexes, encoding='utf-8') as f:
            self.index_file = f.readlines()
        self.len = len(self.index_file)
        self.enhance_cfg = enhance_cfg
        self.enhance_option = train_plan.enhance if train else False
        self.enhance = EnhancePackage(self.image_shape, enhance_cfg)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """

        :param index:
        :return:
        img: [C,H,W]
        tar: (Label, X1,Y1,X2,Y2)
        """
        ran = Random()

        if ran.random() < self.enhance_cfg["mosaic"]:
            lines = sample(self.index_file, 3)
            lines.append(self.index_file[index])
            shuffle(lines)
            final_img, final_xyxy, label = self.get_mosaic_item(lines)
        else:
            final_img, final_xyxy, label = self.get_single_item(self.index_file[index])

        # Step 3 归一，并最终确认增广后图像尺寸是否符合输入尺寸，如果不符合，在此直接拉升
        final_img = final_img.astype(np.float32) / 255

        # Step 4 调整
        label = torch.from_numpy(label)

        final_xyxy[:, [0, 2]] /= final_img.shape[1]
        final_xyxy[:, [1, 3]] /= final_img.shape[0]

        final_xywh = cvt_bbox(final_xyxy, CvtFlag.CVT_XYXY_XYWH)
        final_xywh = torch.from_numpy(final_xywh)

        # Step 5 拼接最终 [ImageIndex]_[label]_[XYWH]
        labels_out = torch.zeros((label.shape[0], 6))
        if not label.shape[0] != 0:
            labels_out[:, 1] = label
            labels_out[:, 2:] = final_xywh

        return torch.from_numpy(final_img).permute(2, 0, 1), labels_out

    def get_single_item(self, line):
        line = line.split()
        image_file = line[0]

        # Step 1 获取原始图像和标签信息
        img = cv2.imread(image_file)
        label_box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]], dtype=np.float32)
        is_empty = label_box.shape[0] == 0
        label = label_box[..., 4] if not is_empty else np.empty(shape=(0, 1))
        box_xyxy = label_box[..., 0:4] if not is_empty else np.empty(shape=(0, 4))

        # Step 2 图像增广
        img = np.array(img)
        final_img, final_xyxy = self.enhance(img, box_xyxy, self.enhance_option)

        return final_img, final_xyxy, label

    def get_mosaic_item(self, lines):
        for l in lines:
            final_img, final_xyxy, label = self.get_single_item(l)
        return self.get_single_item(lines[0])

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            info += "{}:\t{}\r\n".format(key, value)
        return info


if __name__ == "__main__":

    rank = 1
    plan = TrainPlan("../cfg/raccoon.yaml")
    dataset = YoloDataset(plan, cvt_cfg(plan.enhance_cfg))
    dataloader = InfiniteDataLoader(dataset, batch_size=1,
                                    shuffle=False,
                                    collate_fn=collate_fn)

    pbar = tqdm(dataloader)
    for images, targets in pbar:
        pass
    pbar.close()
