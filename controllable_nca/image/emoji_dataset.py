import matplotlib.pyplot as plt
import torch
from einops import rearrange

from controllable_nca.dataset import MultiClass2DDataset
from controllable_nca.utils import load_emoji, rgb


class EmojiDataset(MultiClass2DDataset):
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

    def __init__(self, image_size=64):
        emojis = torch.stack(
            [load_emoji(e, image_size) for e in EmojiDataset.EMOJI], dim=0
        )
        targets = torch.arange(emojis.size(0))
        super(EmojiDataset, self).__init__(emojis, targets)
        self.digits = torch.stack(
            [load_emoji(None, image_size, code=e) for e in EmojiDataset.digits], dim=0
        )

    def visualize(self, idx=0):
        self.plot_img(self.x[idx : idx + 1])

    def plot_img(self, img):
        with torch.no_grad():
            rgb_image = rgb(img, False).squeeze().detach().cpu().numpy()
        rgb_image = rearrange(rgb_image, "c w h -> w h c")
        _ = plt.imshow(rgb_image)
        plt.show()
