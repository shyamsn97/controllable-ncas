import matplotlib.pyplot as plt
import torch
from einops import rearrange

from controllable_nca.dataset import NCADataset
from controllable_nca.utils import load_emoji, rgb


def plot_img(img):
    with torch.no_grad():
        rgb_image = rgb(img, False).squeeze().detach().cpu().numpy()
    rgb_image = rearrange(rgb_image, "c w h -> w h c")
    _ = plt.imshow(rgb_image)
    plt.show()


def draw_in_grid(img, x=None, y=None, grid_size=64):
    with torch.no_grad():
        if x is None:
            x = grid_size // 2
        if y is None:
            y = grid_size // 2

        img_size = img.size(-1)
        center = img_size // 2
        grid = torch.zeros(1, img.size(1), grid_size, grid_size, device=img.device)

        min_x = x - center
        min_x_diff = 0 - min(0, min_x)
        max_x = x + center
        max_x_diff = grid_size - max(64, max_x)
        min_x = min_x + max_x_diff + min_x_diff
        max_x = max_x + max_x_diff + min_x_diff

        min_y = y - center
        min_y_diff = 0 - min(0, min_y)
        max_y = y + center
        max_y_diff = grid_size - max(grid_size, max_y)
        min_y = min_y + max_y_diff + min_y_diff
        max_y = max_y + max_y_diff + min_y_diff

        grid[:, :, min_x:max_x, min_y:max_y] += img
        return grid, (min_x + center, min_y + center)


class MovableEmojiDataset(NCADataset):
    def __init__(self, emoji: str = "🦎", grid_size=64, image_size=32):
        super().__init__()
        self.grid_size = grid_size
        self.image_size = image_size
        self.emoji = emoji
        self.x = load_emoji(emoji, image_size).unsqueeze(0)

    def draw(self, x=None, y=None, substrate=None):
        if substrate is None:
            substrate = self.x
        return draw_in_grid(substrate.clone(), x=x, y=y, grid_size=self.grid_size)

    def target_size(self):
        return (4, self.grid_size, self.grid_size)

    def visualize(self, x=None, y=None):
        grid, coords = self.draw(x, y)
        plot_img(grid)

    def to(self, device: torch.device):
        self.x = self.x.to(device)

    def num_goals(self) -> int:
        return 5
