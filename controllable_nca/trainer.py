import math
import random
from datetime import datetime
from typing import Any, Optional, Tuple  # noqa

import torch
import tqdm
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
        self.num_damaged = num_damaged

        self.log_base_path = log_base_path
        self.log_path = "{}/{}".format(log_base_path, datetime.now())
        print("Writing to {}".format(self.log_path))
        self.train_writer = SummaryWriter(self.log_path, flush_secs=10)
        self.sort_loss = self.loss
        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")

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

    def train(self, batch_size, epochs, *args, **kwargs):
        self.pool = SamplePool(self.pool_size)
        bar = tqdm.tqdm(range(epochs))
        for i in bar:
            idxs = random.sample(range(len(self.pool)), batch_size)
            batch = self.sample_batch(idxs, self.pool)
            targets = self.sample_targets(idxs)
            # Sort by loss, descending.
            with torch.no_grad():
                sort_idx = torch.argsort(
                    self.sort_loss(batch, targets), descending=True
                )
                batch = batch[sort_idx]
                # Replace the highest-loss sample with the seed.
                batch[0] = self.nca.generate_seed(1)[0].to(self.device)
                if self.num_damaged > 0:
                    batch = self.damage(batch)
            batch.requires_grad = True
            # Perform training.
            outputs, loss, metrics = self.train_batch(batch, targets)
            # Place outputs back in the pool.
            self.update_pool(idxs, outputs, targets)

            bar.set_description(
                "loss: {} -- log10_loss: {}".format(loss, math.log10(loss))
            )
            self.emit_metrics(i, batch, outputs, loss, metrics={})

    def visualize(self, *args, **kwargs):
        raise NotImplementedError("Visualize is not implemented!")
