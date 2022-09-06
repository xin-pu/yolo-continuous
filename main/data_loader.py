import torch

from cfg.train_plan import TrainPlan
from dataset.infinite_dataLoader import InfiniteDataLoader
from dataset.yolo_dataset import YoloDataset, collate_fn
from dataset.yolo_dataset_git import YoloDataset2, yolo_dataset_collate
from utils.helper_io import cvt_cfg, check_file


def get_dataloader(train_plan: TrainPlan, train=True):
    enhance_cfg = cvt_cfg(train_plan.enhance_cfg)
    with open(train_plan.train_indexes if train else train_plan.val_indexes, encoding='utf-8') as f:
        index_file = f.readlines()
    dataset = YoloDataset2(index_file, input_shape=[640, 640],
                           num_classes=1,
                           anchors=train_plan.anchors,
                           anchors_mask=train_plan.anchors_mask,
                           epoch_length=train_plan.epochs,
                           mosaic=True,
                           mixup=True,
                           mosaic_prob=0.5,
                           mixup_prob=0.5,
                           train=train)
    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=train_plan.batch_size,
                                    shuffle=train_plan.shuffle,
                                    num_workers=train_plan.workers,
                                    pin_memory=train_plan.pin_memory,
                                    drop_last=train_plan.drop_last,
                                    collate_fn=yolo_dataset_collate, )
    return dataloader


if __name__ == "__main__":
    _train_cfg_file = check_file(r"../cfg/voc_train.yaml")
    _train_plan = TrainPlan(_train_cfg_file)
    dl = get_dataloader(_train_plan, False)
    for a, b in dl:
        print(b)
        break
    torch.cuda.empty_cache()
