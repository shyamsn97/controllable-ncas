from datetime import datetime
from typing import Optional, Tuple  # noqa

import torch

from controllable_nca.experiments.movable_emoji.movable_emoji_dataset import (
    MovableEmojiDataset,
)
from controllable_nca.experiments.movable_emoji.trainer import MovableEmojiNCATrainer
from controllable_nca.nca import ControllableNCA

if __name__ == "__main__":
    dataset = MovableEmojiDataset(grid_size=64, image_size=32)

    NUM_HIDDEN_CHANNELS = 32

    nca = ControllableNCA(
        num_goals=dataset.num_goals(),
        target_shape=dataset.target_size(),
        living_channel_dim=3,
        num_hidden_channels=NUM_HIDDEN_CHANNELS,
        cell_fire_rate=1.0,
    )

    device = torch.device("cuda")
    nca = nca.to(device)
    dataset.to(device)

    trainer = MovableEmojiNCATrainer(
        nca,
        dataset,
        nca_steps=[48, 64],
        lr=1e-3,
        num_damaged=0,
        damage_radius=3,
        device=device,
        pool_size=256,
    )

    try:
        trainer.train(batch_size=24, epochs=100000)
    except KeyboardInterrupt:
        nca.save("MovableEmoji_{}.pt".format(datetime.now()))
