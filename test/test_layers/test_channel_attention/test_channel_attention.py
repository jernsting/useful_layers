import unittest
import torch
import numpy as np

import useful_layers as ul


class ChannelAttention2DTest(unittest.TestCase):

    def test_channel_attention_layer_shape(self):
        se_layer = ul.layers.ChannelAttention2D(3)

        dummy_input = torch.randn(2, 3, 15, 15)  # b, c, h, w
        output = se_layer(dummy_input)

        dummy_input = dummy_input.detach().numpy()
        output = output.detach().numpy()

        self.assertFalse(np.array_equal(dummy_input, dummy_input * output),
                         'The dummy input should be changed by the layer')
        self.assertListEqual([dummy_input.shape[0], dummy_input.shape[1], 1, 1], list(output.shape))


class ChannelAttention3DTest(unittest.TestCase):

    def test_channel_attention_layer_shape(self):
        se_layer = ul.layers.ChannelAttention3D(3)

        dummy_input = torch.randn(2, 3, 15, 15, 15)  # b, c, d, h, w
        output = se_layer(dummy_input).detach().numpy()

        dummy_input = dummy_input.detach().numpy()

        self.assertFalse(np.array_equal(dummy_input, dummy_input * output),
                         'The dummy input should be changed by the layer')
        self.assertListEqual([dummy_input.shape[0], dummy_input.shape[1], 1, 1, 1], list(output.shape),
                             'The shape of input and output should match')


if __name__ == '__main__':
    unittest.main()
