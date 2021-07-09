# Squeeze and Excitation
This module is inspired by the Squeeze and Excitation Normalization module: [github](https://github.com/iantsen/hecktor/) or [arxiv](https://arxiv.org/pdf/2102.10446.pdf)

## Class - SqueezeAndExcitation2D

Simple SqueezeAndExcitation for 2D filters.

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_channels | int | Number of input channels|
| reduction | int, default=2 | Degree of reduction |

#### Example

A simple usage example without context:

```python
import torch
import useful_layers as ul


layer = ul.SqueezeAndExcitation2D(in_channels=5,
                            	  reduction=2)
```


## Class - SqueezeAndExcitation3D

Simple SqueezeAndExcitation for 3D filters.

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_channels | int | Number of input channels|
| reduction | int, default=2 | Degree of reduction |

#### Example

A simple usage example without context:

```python
import torch
import useful_layers as ul


layer = ul.SqueezeAndExcitation3D(in_channels=5,
                            	  reduction=2)
```
