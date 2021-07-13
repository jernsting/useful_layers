# useful_layers Extensions

Extending the useful_layers packet with own layers is straightforward:

## Adding Layers
To add a new layer implementation simply inherit from `useful_layers.Layer`.

Implement the `forward(x: torch.Tensor) -> torch.Tensor:` function just as you would do with vanilla pytorch.

## Adding Blocks
To add a new block your implementation has to inherit from `useful_layers.Block`.

Implement the `forward(x: torch.Tensor) -> torch.Tensor:` function just as you would do with vanilla pytorch.