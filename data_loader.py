import torch

from dataset.images_and_labels import ImagesAndLabels
from dataset.infinite_dataLoader import InfiniteDataLoader
from utils.helper_io import cvt_cfg, check_file


def get_dataloader(train_cfg, train=True):
    enhance_cfg = cvt_cfg(train_cfg['enhance_cfg'])
    dataset = ImagesAndLabels(train_cfg, enhance_cfg, train)
    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=train_cfg['batch_size'],
                                    shuffle=train_cfg['shuffle'],
                                    pin_memory=True,
                                    num_workers=1,
                                    collate_fn=ImagesAndLabels.collate_fn)
    return dataloader


if __name__ == "__main__":
    train_cfg_file = check_file(r"cfg/voc_train.yaml")
    dl = get_dataloader(cvt_cfg(train_cfg_file), False)
    for a, b in dl:
        pass
    torch.cuda.empty_cache()
