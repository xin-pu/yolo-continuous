import torch
from torch.utils.data import DataLoader

from dataset.infinite_dataLoader import InfiniteDataLoader
from dataset.yolo_dataset import YoloDataset
from utils.helper_io import cvt_cfg, check_file


def collate_fn(batch):
    img, label = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0)


def get_dataloader(train_cfg, train=True):
    enhance_cfg = cvt_cfg(train_cfg['enhance_cfg'])
    dataset = YoloDataset(train_cfg, enhance_cfg, train)
    dataloader = DataLoader(dataset,
                            batch_size=train_cfg['batch_size'],
                            shuffle=train_cfg['shuffle'],
                            num_workers=train_cfg['workers'],
                            pin_memory=True,
                            collate_fn=collate_fn,
                            drop_last=True)
    return dataloader


if __name__ == "__main__":
    train_cfg_file = check_file(r"../cfg/voc_train.yaml")
    dl = get_dataloader(cvt_cfg(train_cfg_file), False)
    for a, b in dl:
        print(b)
        break
    torch.cuda.empty_cache()
