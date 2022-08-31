import torch
from dataset.infinite_dataLoader import InfiniteDataLoader
from dataset.yolo_dataset import YoloDataset, collate_fn
from utils.helper_io import cvt_cfg, check_file


def get_dataloader(train_cfg, train=True):
    enhance_cfg = cvt_cfg(train_cfg['enhance_cfg'])
    dataset = YoloDataset(train_cfg, enhance_cfg, train)
    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=train_cfg['batch_size'],
                                    shuffle=train_cfg['shuffle'],
                                    num_workers=train_cfg['workers'],
                                    pin_memory=train_cfg['pin_memory'],
                                    drop_last=train_cfg['drop_last'],
                                    collate_fn=collate_fn, )
    return dataloader


if __name__ == "__main__":
    train_cfg_file = check_file(r"../cfg/voc_train.yaml")
    dl = get_dataloader(cvt_cfg(train_cfg_file), False)
    for a, b in dl:
        print(b)
        break
    torch.cuda.empty_cache()
