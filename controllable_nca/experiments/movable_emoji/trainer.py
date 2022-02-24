import math
import random
from typing import Any, Optional, Tuple  # noqa

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from controllable_nca.dataset import NCADataset
from controllable_nca.nca import ControllableNCA
from controllable_nca.sample_pool import SamplePool
from controllable_nca.trainer import NCATrainer


class MovableEmojiNCATrainer(NCATrainer):
    def __init__(
        self,
        nca: ControllableNCA,
        target_dataset: NCADataset,
        nca_steps=[48, 64],
        lr: float = 2e-3,
        pool_size: int = 512,
        num_damaged: int = 0,
        log_base_path: str = "tensorboard_logs",
        damage_radius: int = 3,
        device: Optional[torch.device] = None,
    ):
        super(MovableEmojiNCATrainer, self).__init__(
            pool_size, num_damaged, log_base_path, device
        )
        self.target_dataset = target_dataset
        self.target_size = self.target_dataset.target_size()

        self.nca = nca
        self.min_steps = nca_steps[0]
        self.max_steps = nca_steps[1]

        self.num_target_channels = self.target_size[0]
        self.image_size = self.target_size[-1]
        self.rgb = self.target_size[0] == 3
        self.damage_radius = damage_radius

        self.optimizer = torch.optim.Adam(self.nca.parameters(), lr=lr)
        self.lr_sched = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, [5000], 0.3
        )

    def loss(self, x, targets):
        return F.mse_loss(
            x[:, : self.num_target_channels, :, :],
            targets[:, : self.num_target_channels, :, :],
            reduction="none",
        ).mean(dim=(1, 2, 3))

    def sample_batch(
        self, sampled_indices, sample_pool=None, replace=2
    ) -> Tuple[Any, Any]:
        """
        Returns batch + targets

        Returns:
            Tuple[Any, Any]: [description]
        """
        default_coords = (
            self.target_dataset.grid_size // 2,
            self.target_dataset.grid_size // 2,
        )
        if sample_pool is not None:
            batch = sample_pool[sampled_indices]
        else:
            batch = [None for _ in range(len(sampled_indices))]
        for i in range(len(sampled_indices)):
            if batch[i] is None:
                batch[i] = (
                    self.nca.generate_seed(1)[0].to(self.device),
                    default_coords,
                )
            elif torch.sum(self.nca.alive(batch[i][0].unsqueeze(0))) == 0.0:
                batch[i] = (
                    self.nca.generate_seed(1)[0].to(self.device),
                    default_coords,
                )
        for i in range(replace):
            batch[-i] = (self.nca.generate_seed(1)[0].to(self.device), default_coords)
        return batch

    def sample_targets(self, batch_coords, directions, steps):
        out_targets = []
        goals = []
        out_coords = []

        for coords, direction, num_steps in zip(batch_coords, directions, steps):
            if direction == 0:
                # stay in place
                new_coords = coords
            if direction == 1:
                # left
                new_coords = (coords[0] - (num_steps // 8), coords[1])
            if direction == 2:
                # right
                new_coords = (coords[0] + (num_steps // 8), coords[1])
            if direction == 3:
                # up
                new_coords = (coords[0], coords[1] - (num_steps // 8))
            if direction == 4:
                # down
                new_coords = (coords[0], coords[1] + (num_steps // 8))

            target, new_coords = self.target_dataset.draw(new_coords[0], new_coords[1])

            goals.append(torch.tensor(direction, device=self.device))
            out_coords.append(new_coords)
            out_targets.append(target)

        return out_coords, out_targets, goals

    def train_batch(self, batch, directions, num_steps):
        coords = []
        substrates = []
        steps = [num_steps] * len(batch)
        for i in range(len(batch)):
            substrate, batch_coords = batch[i]
            substrates.append(substrate.to(self.device))
            coords.append(batch_coords)

        new_coords, targets, goals = self.sample_targets(coords, directions, steps)

        substrates = torch.stack(substrates, dim=0).squeeze()
        targets = torch.stack(targets, dim=0).squeeze().to(self.device)
        if self.nca.use_image_encoder:
            goals = targets
        else:
            goals = torch.stack(goals, dim=0).squeeze().to(self.device)
        substrates = self.nca.grow(substrates, num_steps=num_steps, goal=goals)

        loss = self.loss(substrates, targets).mean()
        self.optimizer.zero_grad()
        loss.backward()
        for p in self.nca.parameters():
            if p.grad is not None:
                p.grad /= torch.norm(p.grad) + 1e-10
        self.optimizer.step()
        grad_dict = {}
        for n, W in self.nca.named_parameters():
            if W.grad is not None:
                grad_dict["{}_grad".format(n)] = float(torch.sum(W.grad).item())

        return (
            substrates.detach(),
            list(zip(list(substrates.detach().cpu()), new_coords)),
            targets,
            loss.item(),
            {"loss": loss.item(), "log10loss": math.log10(loss.item()), **grad_dict},
        )

    def train(self, batch_size, epochs):
        bar = tqdm.tqdm(range(epochs))
        self.pool = SamplePool(self.pool_size)

        for i in bar:
            idxs = random.sample(range(self.pool_size), batch_size)

            with torch.no_grad():
                batch = self.sample_batch(idxs, self.pool, replace=2)

            # train center
            directions = [0] * batch_size
            num_steps = np.random.randint(self.min_steps, self.max_steps)
            substrates, new_batch, center_targets, loss, metrics = self.train_batch(
                batch, directions, num_steps
            )

            # random directions
            directions = np.random.randint(1, 5, batch_size)

            # take small steps
            (
                substrates,
                small_steps_batch,
                small_targets,
                loss,
                metrics,
            ) = self.train_batch(new_batch, directions, self.max_steps // 2)

            # take more steps
            substrates, med_steps_batch, med_targets, loss, metrics = self.train_batch(
                small_steps_batch, directions, self.max_steps // 2
            )

            # take more steps
            (
                substrates,
                large_steps_batch,
                large_targets,
                loss,
                metrics,
            ) = self.train_batch(med_steps_batch, directions, self.max_steps // 2)

            self.pool[idxs] = small_steps_batch
            self.lr_sched.step()

            description = "--".join(["{}:{}".format(k, metrics[k]) for k in metrics])
            bar.set_description(description)
            outputs = [
                ("sampled_batch", batch, None),
                ("centered", new_batch, center_targets),
                ("small", small_steps_batch, small_targets),
                ("med", med_steps_batch, med_targets),
                ("large", large_steps_batch, large_targets),
            ]
            self.emit_metrics(i, outputs, metrics)

    def emit_metrics(self, i: int, outputs, metrics={}):
        with torch.no_grad():
            for o in outputs:
                keys, batch, targets = o
                batch = [b[0].detach().cpu() for b in batch]
                batch = torch.stack(batch, dim=0)
                self.train_writer.add_images(
                    keys + "_batch", self.to_rgb(batch), i, dataformats="NCHW"
                )
                if targets is not None:
                    self.train_writer.add_images(
                        keys + "_targets", self.to_rgb(targets), i, dataformats="NCHW"
                    )

            for k in metrics:
                self.train_writer.add_scalar(k, metrics[k], i)
