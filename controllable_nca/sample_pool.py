from typing import Any, List, Tuple  # noqa

from torch.utils.data import Dataset


class SamplePool(Dataset):
    def __init__(self, pool_size: int = 256):
        self.pool_size = pool_size
        self.pool = [None for _ in range(pool_size)]

    def __len__(self):
        return self.pool_size

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.pool[idx]
        return [self.pool[i] for i in idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, int):
            self.pool[idx] = value
        else:
            for i in range(len(idx)):
                index = idx[i]
                self.pool[index] = value[i]
