import torch

from cfg.train_plan import TrainPlan
from dataset.infinite_dataLoader import InfiniteDataLoader
from dataset.yolo_dataset import YoloDataset, collate_fn
from utils.helper_io import cvt_cfg, check_file


def get_dataloader(train_plan: TrainPlan, train=True):
    enhance_cfg = cvt_cfg(train_plan.enhance_cfg)
    dataset = YoloDataset(train_plan, enhance_cfg, train)
    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=train_plan.batch_size,
                                    shuffle=train_plan.shuffle,
                                    num_workers=train_plan.workers,
                                    pin_memory=train_plan.pin_memory,
                                    drop_last=train_plan.drop_last,
                                    collate_fn=collate_fn, )
    return dataloader


if __name__ == "__main__":
    _train_cfg_file = check_file(r"../cfg/voc_train.yaml")
    _train_plan = TrainPlan(_train_cfg_file)
    dl = get_dataloader(_train_plan, False)
    for a, b in dl:
        print(b)
        break
    torch.cuda.empty_cache()
