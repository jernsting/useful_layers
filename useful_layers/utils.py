import torch.nn as nn
from torch.nn import functional as F

__all__ = ['reduction_network']


def reduction_network(in_channels: int,
                      reduction: int = 2,
                      dim: str = None,)\
        -> (nn.Module, nn.Module):
    """

    """
    if dim not in ["2d", "3d"]:
        raise ValueError(f'dim should be 2d or 3d. Got {dim}')
    if dim == "2d":
        Block = nn.Conv2d
    if dim == "3d":
        Block = nn.Conv3d
    conv1 = Block(in_channels=in_channels,
                  out_channels=in_channels // reduction,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=True)
    conv2 = Block(in_channels=in_channels // reduction,
                  out_channels=in_channels,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=True)
    return conv1, conv2

