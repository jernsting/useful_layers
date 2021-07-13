# Spatial Attention
Basic channel attention as presented in [arxiv](https://arxiv.org/pdf/1807.06521v2.pdf)

## Class - SpatialAttention2D

Simple SpatialAttention for 2D filters.

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_channels | int | Number of input channels|
| kernel_size | int, default=7 | Filter size vor CNN |
| batch_norm  | bool, default=True | If true batch normalization is applied |

#### Return Value
The returned value is the spatial attention map. 
This map is not applied to the original input in any way.

#### Example

A simple usage example without context:

```python
import torch
import useful_layers as ul


layer = ul.layers.SpatialAttention2D(in_channels=5,
                                     kernel_size=2)
```


## Class - SpatialAttention3D

Simple SpatialAttention for 3D filters.

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_channels | int | Number of input channels|
| kernel_size | int, default=7 | Filter size for CNN |
| batch_norm  | bool, default=True | If true batch normalization is applied |

#### Return Value
The returned value is the spatial attention map. 
This map is not applied to the original input in any way.

#### Example

A simple usage example without context:

```python
import torch
import useful_layers as ul


layer = ul.layers.SpatialAttention3D(in_channels=5,
                                     kernel_size=2)
```
