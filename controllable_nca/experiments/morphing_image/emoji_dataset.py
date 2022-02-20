import matplotlib.pyplot as plt
import torch
from einops import rearrange

from controllable_nca.dataset import NCADataset
from controllable_nca.utils import load_emoji, rgb


class EmojiDataset(NCADataset):
    # EMOJI = '🦎😀💥'
    EMOJI = "🦎😀💥👁🐠🦋🐞🕸🥨🎄"

    digits = [
        "0030",  # 0
        "0031",  # 1
        "0032",  # 2
        "0033",  # 3
        "0034",  # 4
        "0035",  # 5
        "0036",  # 6
        "0037",  # 7
        "0038",  # 8
        "0039",  # 9
    ]

    def __init__(self, image_size=64, thumbnail_size=40, use_one_hot: bool = False):
        emojis = torch.stack(
            [load_emoji(e, image_size, thumbnail_size) for e in EmojiDataset.EMOJI],
            dim=0,
        )
        self.emojis = emojis
        self.num_samples = len(self)
        self._target_size = self.emojis.size()[-3:]

    def num_goals(self):
        return self.emojis.size(0)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.emojis[idx : idx + 1], idx
        return self.emojis[idx], idx

    def __len__(self):
        return self.emojis.size(0)

    def target_size(self):
        if self._target_size is not None:
            return self._target_size
        self._target_size = self.emojis.size()[-3:]
        return self._target_size

    def to(self, device: torch.device):
        self.emojis = self.emojis.to(device)

    def visualize(self, idx=0):
        self.plot_img(self.emojis[idx : idx + 1])

    def plot_img(self, img):
        with torch.no_grad():
            rgb_image = rgb(img, False).squeeze().detach().cpu().numpy()
        rgb_image = rearrange(rgb_image, "c w h -> w h c")
        _ = plt.imshow(rgb_image)
        plt.show()
