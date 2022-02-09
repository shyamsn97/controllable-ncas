import abc

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NCADataset(Dataset, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def target_size(self):
        """
        returns size of single target instance
        """

    def to(self, device: torch.device):
        pass

    def visualize(self):
        pass


class MultiClass2DDataset(NCADataset):
    def __init__(
        self,
        x: torch.tensor,
        y: torch.tensor,
        use_one_hot: bool = True,
        target_size=None,
    ):
        self.x = x
        self.y = y
        self.use_one_hot = use_one_hot
        self.num_classes = len(torch.unique(y))
        self.one_hot = None
        if self.use_one_hot:
            self.one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        self.num_samples = len(self)
        self._target_size = target_size

    def __getitem__(self, idx):
        if isinstance(idx, int):
            out = None
            if self.one_hot is not None:
                out = self.one_hot[idx : idx + 1]
            return self.x[idx : idx + 1], out
        out = None
        if self.one_hot is not None:
            out = self.one_hot[idx]
        return self.x[idx], out

    def __len__(self):
        return self.x.size(0)

    def target_size(self):
        if self._target_size is not None:
            return self._target_size
        self._target_size = self.x.size()[-3:]
        return self._target_size

    def to(self, device: torch.device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        if self.one_hot is not None:
            self.one_hot = self.one_hot.to(device)
