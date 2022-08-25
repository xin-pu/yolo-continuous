import cv2
import numpy as np
import torch.distributed
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from dataset.infinite_dataLoader import InfiniteDataLoader
from image_enhance.enhance_package import EnhancePackage
from utils.bbox import cvt_bbox, CvtFlag
from utils.helper_io import cvt_cfg


class ImagesAndLabels(Dataset):

    def __init__(self,
                 data_cfg,
                 enhance_cfg,
                 train=True):

        self.image_shape = (data_cfg["image_size"], data_cfg["image_size"])
        with open(data_cfg["train"] if train else data_cfg["val"], encoding='utf-8') as f:
            self.index_file = f.readlines()
        self.len = len(self.index_file)

        self.enhance_option = data_cfg["enhance"] if train else False
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
        line = self.index_file[index].split()
        image_file = line[0]

        # Step 1 获取原始图像和标签信息
        img = cv2.imread(image_file)
        label_box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]], dtype=np.float32)
        is_empty = label_box.shape[0] == 0
        label = label_box[..., 0] if not is_empty else np.empty(shape=(0, 1))
        box_xyxy = label_box[..., 1:] if not is_empty else np.empty(shape=(0, 4))

        # Step 2 图像增广
        img = np.array(img)
        final_img, final_xyxy = self.enhance(img, box_xyxy, self.enhance_option)

        # Step 3 归一，并最终确认增广后图像尺寸是否符合输入尺寸，如果不符合，在此直接拉升
        final_img = final_img.astype(np.float32) / 255

        # Step 4 调整
        label = torch.from_numpy(label)

        final_xywh = cvt_bbox(final_xyxy, CvtFlag.CVT_XYXY_XYWH)  #
        final_xywh = torch.from_numpy(final_xywh)
        final_xywh[:, [1, 3]] /= final_img.shape[0]
        final_xywh[:, [0, 2]] /= final_img.shape[1]

        # Step 5 拼接最终 [ImageIndex]_[label]_[XYWH]
        labels_out = torch.zeros((label_box.shape[0], 6))
        if not is_empty:
            labels_out[:, 1] = label
            labels_out[:, 2:] = final_xywh

        return torch.from_numpy(final_img).permute(2, 0, 1), labels_out

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            info += "{}:\t{}\r\n".format(key, value)
        return info

    # Keypoint 数据裁剪函数，其实将一个batch下不同数量的Label,Box 以序号做索引拼接成一个张量
    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0)


if __name__ == "__main__":
    _data_cfg = cvt_cfg("../cfg/voc_train.yaml")
    _enhance_cfg = cvt_cfg("../cfg/enhance/enhance.yaml")
    rank = 1
    dataset = ImagesAndLabels(_data_cfg, _enhance_cfg)
    dataloader = InfiniteDataLoader(dataset, batch_size=_data_cfg["batch_size"],
                                    shuffle=False,
                                    collate_fn=ImagesAndLabels.collate_fn)

    pbar = tqdm(dataloader)
    for images, targets in pbar:
        pass
    pbar.close()
