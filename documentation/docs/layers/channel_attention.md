# Channel Attention
Basic channel attention as presented in [arxiv](https://arxiv.org/pdf/1807.06521v2.pdf)

## Class - ChannelAttention2D

Simple ChannelAttention for 2D filters.

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_channels | int | Number of input channels|
| reduction | int, default=2 | Degree of reduction |

#### Return Value
The returned value is the channel attention map. 
This map is not applied to the original input in any way.

#### Example

A simple usage example without context:

```python
import torch
import useful_layers as ul


layer = ul.layers.ChannelAttention2D(in_channels=5,
                                     reduction=2)
```


## Class - ChannelAttention3D

Simple ChannelAttention for 3D filters.

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_channels | int | Number of input channels|
| reduction | int, default=2 | Degree of reduction |

#### Return Value
The returned value is the channel attention map. 
This map is not applied to the original input in any way.

#### Example

A simple usage example without context:

```python
import torch
import useful_layers as ul


layer = ul.layers.ChannelAttention3D(in_channels=5,
                                     reduction=2)
```
