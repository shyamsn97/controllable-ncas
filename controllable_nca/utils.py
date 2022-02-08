import io
from typing import Iterable, Tuple  # noqa

import numpy as np
import PIL.Image
import requests
import torch
import torch.nn as nn
from einops import rearrange  # noqa


def load_image(url, size):
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((40, 40), PIL.Image.ANTIALIAS)
    img = np.float32(img) / 255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    # pad to self.h, self.h
    diff = size - 40
    img = torch.tensor(img).permute(2, 0, 1)
    img = torch.nn.functional.pad(
        img, [diff // 2, diff // 2, diff // 2, diff // 2], mode="constant", value=0
    )
    return img


def load_emoji(emoji, size, code=None):
    if code is None:
        code = hex(ord(emoji))[2:].lower()
    url = (
        "https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true"
        % code
    )
    return load_image(url, size)


def to_alpha(x):
    return torch.clip(x[:, 3:4], 0.0, 1.0)


def rgb(x, rgb=False):
    # assume rgb premultiplied by alpha
    if rgb:
        return torch.clip(x[:, :3], 0.0, 1.0)
    rgb, a = x[:, :3], to_alpha(x)
    return torch.clip(1.0 - a + rgb, 0.0, 1.0)


def create_2d_circular_mask(h, w, center=None, radius=3):

    if center is None:  # use the middle of the image
        # center = (int(w / 2), int(h / 2))
        center = (
            np.random.randint(radius + 2, w - (radius + 2)),
            np.random.randint(radius + 2, h - (radius + 2)),
        )
        # center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def create_3d_circular_mask(h, w, d, center=None, radius=3):

    if center is None:  # use the middle of the image
        # center = (int(w / 2), int(h / 2))
        center = (
            np.random.randint(radius + 2, w - (radius + 2)),
            np.random.randint(radius + 2, h - (radius + 2)),
            np.random.randint(radius + 2, d - (radius + 2)),
        )
        # center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(
            center[0], center[1], center[2], w - center[0], h - center[1], d - center[2]
        )

    Y, X, Z = np.ogrid[:h, :w, :d]
    dist_from_center = np.sqrt(
        (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    )

    mask = dist_from_center <= radius
    return mask


def get_conv2d_output_shape(
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
) -> Tuple[int]:
    """
    Gets expected output shape of conv
    from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html:
    H_out: [((H_in + 2 x padding - dilation x (kernel_size - 1) - 1) / (stride)) + 1]
    W_out: [((W_in + 2 x padding - dilation x (kernel_size - 1) - 1) / (stride)) + 1]
    Args:
        in_channels (int):
        out_channels (int):
        height (int):
        width (int):
        kernel_size (int):
        padding (int, optional): . Defaults to 0.
        stride (int, optional): . Defaults to 1.
        dilation (int, optional): . Defaults to 1.
    Returns:
        Tuple[int]: (C_out, H_out, W_in)
    """

    calculate_out_size = lambda x: int(
        ((x + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    )  # noqa
    return (out_channels, calculate_out_size(height), calculate_out_size(width))


def build_conv2d_net(
    in_channels: int,
    out_channels: int,
    in_size: int,
    out_size: int,
    hidden_dims: int = 32,
    kernel_size: int = 3,
    stride: int = 2,
    padding: int = 1,
    bias: bool = True,
    activation: torch.nn.Module = torch.nn.ReLU,
) -> torch.nn.Module:
    if in_size == out_size:
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    layers = []
    curr_channel_dims = in_channels
    running_size = in_size
    while running_size > out_size:
        size = get_conv2d_output_shape(
            curr_channel_dims,
            hidden_dims,
            running_size,
            running_size,
            kernel_size,
            padding=padding,
            stride=stride,
        )
        running_size = running_size // 2
        if running_size == out_size:
            layers.append(
                nn.Conv2d(
                    curr_channel_dims,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
            break

        layers.append(
            nn.Conv2d(
                curr_channel_dims,
                hidden_dims,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )
        )
        layers.append(activation())

        curr_channel_dims = size[0]
    return nn.Sequential(*layers)  # remove last relu
