from torch.utils.data.dataloader import DataLoader


class InfiniteDataLoader(DataLoader):
    """
    重用的数据加载器
    使用与普通DataLoader相同的语法
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sampler = RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class RepeatSampler(object):
    """
    重复采样器
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


if __name__ == "__main__":
    pass
