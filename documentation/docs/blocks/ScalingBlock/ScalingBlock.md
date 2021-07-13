# Scaling Block
This block scales (i.e. multiplies) the original input with the activation map.

## Class - ScalingBlock

Simple scaling block.

| Parameter | type | Description |
| -----     | ----- | ----- |
| input_layer | useful_layers.layer | Layer implementation to calculate activation map |

#### Return Value
The returned value is the product of the original input and the output of the layer.

#### Example

A simple usage example without context:

```python
import torch
import useful_layers as ul


layer = ul.layers.ChannelAttention2D(in_channels=5,
                                     reduction=2)
block = ul.blocks.ScalingBlock(layer)
```