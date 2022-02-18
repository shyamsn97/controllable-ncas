import math
from datetime import datetime
from typing import Any, Optional, Tuple  # noqa

import torch
from torch.utils.tensorboard import SummaryWriter

from controllable_nca.sample_pool import SamplePool


class NCATrainer:
    def __init__(
        self,
        pool_size: int = 256,
        num_damaged: int = 0,
        log_base_path="tensorboard_logs",
        device: Optional[torch.device] = None,
    ):
        self.pool_size = pool_size
        self.pool = SamplePool(self.pool_size)
        self.num_damaged = num_damaged

        self.log_base_path = log_base_path
        self.log_path = "{}/{}".format(log_base_path, datetime.now())
        print("Writing to {}".format(self.log_path))
        self.train_writer = SummaryWriter(self.log_path, flush_secs=10)
        self.sort_loss = self.loss
        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")

    def to_alpha(self, x):
        return torch.clamp(x[:, 3:4, :, :], 0.0, 1.0)  # 1.0

    def to_rgb(self, x):
        # assume rgb premultiplied by alpha
        if self.rgb:
            return torch.clamp(x[:, :3], 0.0, 1.0).detach().cpu().numpy()
        rgb = x[:, :3, :, :]  # 0,0,0
        a = self.to_alpha(x)  # 1.0
        im = 1.0 - a + rgb  # (1-1+0) = 0, (1-0+0) = 1
        im = torch.clamp(im, 0, 1)
        return im.detach().cpu().numpy()

    def sample_batch(self, sampled_indices, sample_pool) -> Tuple[Any, Any]:
        """
        Returns batch + targets

        Returns:
            Tuple[Any, Any]: [description]
        """
        raise NotImplementedError("Sampled batch is not implemented!")

    def sample_targets(self, sampled_indices):
        raise NotImplementedError("Sampled targets not implemented!")

    def damage(self, batch):
        return batch

    def emit_metrics(self, i: int, batch, outputs, loss, metrics={}):
        with torch.no_grad():
            self.train_writer.add_scalar("loss", loss, i)
            self.train_writer.add_scalar("log10(loss)", math.log10(loss), i)

    def loss(self, batch, targets):
        raise NotImplementedError("loss not implemented!")

    def train_batch(self, batch, targets) -> Tuple[Any, Any]:
        """
        Single training pass using sampled batch and targets

        Args:
            batch ([type]): [description]
            targets ([type]): [description]

        Returns:
            outputs: Any : Optional outputs returned after train_batch
            loss: Any : loss value
            metrics
        """
        raise NotImplementedError("train_batch not implemented!")

    def update_pool(self, idxs, outputs, targets):
        self.pool[idxs] = outputs

    def visualize(self, *args, **kwargs):
        raise NotImplementedError("Visualize is not implemented!")
