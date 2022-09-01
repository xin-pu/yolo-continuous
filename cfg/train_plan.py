"""
Author: Xin.PU
Email: Pu.Xin@outlook.com
Time: 2022/9/1 10:26
"""
import numpy as np
import yaml


class TrainPlan(object):

    def __init__(self, cfg_file):
        self.cfg_file = cfg = self.get_dataset_cfg(cfg_file)

        self.device = cfg['device']

        # 数据集信息
        self.train_indexes = cfg['train']
        self.val_indexes = cfg['val']
        self.image_size = cfg['image_size']
        self.image_chan = cfg['image_chan']
        self.labels = cfg['labels']
        self.num_labels = len(self.labels)
        self.epochs = cfg['epochs']
        self.batch_size = cfg['batch_size']
        self.shuffle = cfg['shuffle']
        self.workers = cfg['workers']
        self.pin_memory = cfg['pin_memory']
        self.drop_last = cfg['pin_memory']

        self.enhance = cfg['enhance']
        self.enhance_cfg = cfg['enhance_cfg']

        # 模型信息
        self.model_cfg = cfg['model_cfg']

        self.anchors = cfg['anchors']
        self.anchors_mask = cfg['anchors_mask']

        # 优化器信息
        self.adam = cfg['adam']
        self.decay = cfg['decay']
        self.learn_initial = cfg['lrI']
        self.learn_final = cfg['lrF']
        self.momentum = cfg['momentum']
        self.weight_decay = cfg['weight_decay']
        self.warmup = cfg['warmup']
        self.warmup_epochs = cfg['warmup_epochs']
        self.warmup_max_iter = cfg['warmup_max_iter']
        self.warmup_momentum = cfg['warmup_momentum']
        self.warmup_bias_lr = cfg['warmup_bias_lr']

        # 保存信息
        self.resume = cfg['resume']
        self.save_dir = cfg['save_dir']
        self.save_name = cfg['save_name']

    @staticmethod
    def get_dataset_cfg(cfg_file):
        with open(cfg_file, 'r') as file:
            cfg = yaml.safe_load(file)
            return cfg

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            if key not in 'cfg_file':
                info += "%20s :\t%s\r\n" % (key, value)
        return info


if __name__ == '__main__':
    plan = TrainPlan('voc_train.yaml')
    print(plan)
