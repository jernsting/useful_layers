import torch
import torch.nn as nn
from torch.nn import functional as F

from useful_layers.utils import reduction_network

__all__ = ['ChannelAttention2D', 'ChannelAttention3D']


class ChannelAttention2D(nn.Module):
    """ChannelAttention2D

    Channel attention layer as presented in
    https://arxiv.org/pdf/1807.06521v2.pdf.
    """

    def __init__(self,
                 in_channels: int,
                 reduction: int = 2):
        """Create ChannelAttention2D Layer

        Args:
            in_channels (int): Number of input channels
            reduction (int, optional): Degree of reduction. Defaults to 2.
        """
        super(ChannelAttention2D, self).__init__()
        self.conv1, self.conv2 = reduction_network(in_channels, reduction, "2d")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        avg_comp = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1)
        max_comp = torch.max(x.view(b, c, -1), dim=-1).values.view(b, c, 1, 1)
        avg_comp = self.conv2(F.relu(self.conv1(avg_comp)))
        max_comp = self.conv2(F.relu(self.conv1(max_comp)))
        return F.sigmoid(avg_comp + max_comp) * x


class ChannelAttention3D(nn.Module):
    """ChannelAttention3D

    Channel attention layer as presented in
    https://arxiv.org/pdf/1807.06521v2.pdf.
    """

    def __init__(self,
                 in_channels: int,
                 reduction: int = 2):
        """Create ChannelAttention3D Layer

        Args:
            in_channels (int): Number of input channels
            reduction (int, optional): Degree of reduction. Defaults to 2.
        """
        super(ChannelAttention3D, self).__init__()
        self.conv1, self.conv2 = reduction_network(in_channels, reduction, "3d")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.size()
        avg_comp = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1, 1)
        max_comp = torch.max(x.view(b, c, -1), dim=-1).values.view(b, c, 1, 1, 1)
        avg_comp = self.conv2(F.relu(self.conv1(avg_comp)))
        max_comp = self.conv2(F.relu(self.conv1(max_comp)))
        return F.sigmoid(avg_comp + max_comp) * x
