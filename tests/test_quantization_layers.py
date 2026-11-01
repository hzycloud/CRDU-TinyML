import unittest

import torch
from torch import nn

from cig_mpq.quantization import quantize_mixed_precision, quantize_selected_layers


class QuantizationLayersTest(unittest.TestCase):
    def test_selected_layer_uses_requested_error_level(self):
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=1),
        )
        original = model[0].weight.detach().clone()

        quantize_selected_layers(model, {"0": 4, "2": 8})

        error4 = torch.mean((model[0].weight - original) ** 2).item()
        self.assertGreater(error4, 0.0)
        self.assertLessEqual(model[0].weight.abs().max().item(), original.abs().max().item())

    def test_mixed_precision_defaults_unlisted_layers_to_w8(self):
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
        )
        original_first = model[0].weight.detach().clone()
        original_second = model[2].weight.detach().clone()

        quantize_mixed_precision(model, {"0": 4}, default_bits=8)

        error_first = torch.mean((model[0].weight - original_first) ** 2).item()
        error_second = torch.mean((model[2].weight - original_second) ** 2).item()
        self.assertGreater(error_first, error_second)


if __name__ == "__main__":
    unittest.main()
