# Squeeze and Excitation
This module is inspired by the Squeeze and Excitation Normalization module: [github](https://github.com/iantsen/hecktor/) or [arxiv](https://arxiv.org/pdf/2102.10446.pdf)

## Class - SqueezeAndExcitation2D

Simple SqueezeAndExcitation for 2D filters.

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_channels | int | Number of input channels|
| reduction | int, default=2 | Degree of reduction |

#### Return Value
The returned value is the squeeze and excitation map. 
This map is not applied to the original input in any way.

#### Example

A simple usage example without context:

```python
import torch
import useful_layers as ul

class demo(torch.nn.Module):
    def __init__(self, in_channels):
        # ...
        self.layer = ul.layers.SqueezeAndExcitation2D(in_channels=in_channels,
                                                      reduction=2)

    def forward(self, x):
        se_map = self.layer(x)   # This map is not applied!
        return x * se_map        # This is the default behaviour as described in the paper
```


## Class - SqueezeAndExcitation3D

Simple SqueezeAndExcitation for 3D filters.

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_channels | int | Number of input channels|
| reduction | int, default=2 | Degree of reduction |

#### Return Value
The returned value is the Squeeze and Excitation map. 
This map is not applied to the original input in any way.

#### Example

A simple usage example without context:

```python
import torch
import useful_layers as ul

class demo(torch.nn.Module):
    def __init__(self, in_channels):
        # ...
        self.layer = ul.layers.SqueezeAndExcitation3D(in_channels=in_channels,
                            	                      reduction=2)

    def forward(self, x):
        se_map = self.layer(x)   # This map is not applied!
        return x * se_map        # This is the default behaviour as described in the paper
```
