from datetime import datetime
from typing import Optional, Tuple  # noqa

import torch
import torch.nn as nn
from torch.nn import Embedding

from controllable_nca.experiments.morphing_image.emoji_dataset import EmojiDataset
from controllable_nca.experiments.morphing_image.trainer import MorphingImageNCATrainer
from controllable_nca.nca import ControllableNCA

if __name__ == "__main__":

    class DeepEncoder(nn.Module):
        def __init__(self, num_embeddings: int, out_channels: int):
            super().__init__()
            self.num_embeddings = num_embeddings
            self.embedding = Embedding(num_embeddings, 32)
            self.encoder = nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, out_channels, bias=False),
            )

        def forward(self, indices):
            embeddings = self.encoder(self.embedding(indices))
            return embeddings

    NUM_HIDDEN_CHANNELS = 16

    dataset = EmojiDataset(image_size=64, thumbnail_size=40)

    encoder = DeepEncoder(dataset.num_goals(), NUM_HIDDEN_CHANNELS)

    nca = ControllableNCA(
        num_goals=dataset.num_goals(),
        use_image_encoder=False,
        encoder=encoder,
        target_shape=dataset.target_size(),
        living_channel_dim=3,
        num_hidden_channels=NUM_HIDDEN_CHANNELS,
        cell_fire_rate=0.5,
    )

    device = torch.device("cuda")
    nca = nca.to(device)
    dataset.to(device)

    trainer = MorphingImageNCATrainer(
        nca,
        dataset,
        nca_steps=[48, 96],
        lr=1e-3,
        num_damaged=0,
        damage_radius=3,
        device=device,
        pool_size=1024,
    )

    try:
        trainer.train(batch_size=8, epochs=100000)
    except KeyboardInterrupt:
        nca.save("MorphingEmoji_{}.pt".format(datetime.now()))
