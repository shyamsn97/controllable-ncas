import abc

import torch
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

    @abc.abstractmethod
    def num_goals(self) -> int:
        pass
