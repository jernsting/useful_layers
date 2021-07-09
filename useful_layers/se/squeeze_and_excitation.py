import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['SqueezeAndExcitation2D', 'SqueezeAndExcitation3D']


class SqueezeAndExcitation2D(nn.Module):
    """SqueezeAndExcitation2D

        Simple squeeze and excitation layer.

        Inspired by https://github.com/iantsen/hecktor/blob/main/src/layers.py
    """

    def __init__(self,
                 in_channels: int,
                 reduction: int = 2) -> None:
        """Create SqueezeAndExcitation2D Layer

        Args:
            in_channels (int): Number of input channels
            reduction (int, optional): Degree of reduction. Defaults to 2.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels // reduction,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels // reduction,
                               out_channels=in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        out = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        return torch.sigmoid(out) * x


class SqueezeAndExcitation3D(nn.Module):
    """SqueezeAndExcitation3D

        Simple squeeze and excitation layer.

        Inspired by https://github.com/iantsen/hecktor/blob/main/src/layers.py
    """

    def __init__(self,
                 in_channels: int,
                 reduction: int = 2) -> None:
        """Create SqueezeAndExcitation3D Layer

        Args:
            in_channels (int): Number of input channels
            reduction (int, optional): Degree of reduction. Defaults to 2.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=in_channels // reduction,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)
        self.conv2 = nn.Conv3d(in_channels=in_channels // reduction,
                               out_channels=in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.size()
        out = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1, 1)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        return torch.sigmoid(out) * x
