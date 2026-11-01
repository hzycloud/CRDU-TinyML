import unittest

import torch
from torch import nn

from cig_mpq.quantization import (
    fake_quantize_activation_per_tensor,
    install_activation_quantization_hooks,
)


class ActivationQuantizationTest(unittest.TestCase):
    def test_activation_quantization_limits_unique_values(self):
        activation = torch.linspace(0.0, 1.0, steps=1000)

        quantized = fake_quantize_activation_per_tensor(activation, bits=4)

        self.assertEqual(quantized.shape, activation.shape)
        self.assertLessEqual(torch.unique(quantized).numel(), 16)
        self.assertGreaterEqual(quantized.min().item(), activation.min().item())
        self.assertLessEqual(quantized.max().item(), activation.max().item())

    def test_hooks_change_output_and_can_be_removed(self):
        model = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1, bias=False))
        with torch.no_grad():
            model[0].weight.fill_(1.0)
        inputs = torch.linspace(-1.0, 1.0, steps=64).reshape(1, 1, 8, 8)
        original = model(inputs)

        handles = install_activation_quantization_hooks(model, bits=4)
        quantized = model(inputs)
        for handle in handles:
            handle.remove()
        restored = model(inputs)

        self.assertEqual(quantized.shape, original.shape)
        self.assertFalse(torch.allclose(quantized, original))
        self.assertTrue(torch.allclose(restored, original))


if __name__ == "__main__":
    unittest.main()
