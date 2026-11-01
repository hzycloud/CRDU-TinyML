import unittest

import torch

from cig_mpq.quantization import fake_quantize_per_channel


class QuantizationTest(unittest.TestCase):
    def test_int8_has_lower_error_than_int4(self):
        weight = torch.linspace(-1.0, 1.0, steps=257).reshape(1, 1, 257)

        q4 = fake_quantize_per_channel(weight, bits=4)
        q8 = fake_quantize_per_channel(weight, bits=8)

        self.assertEqual(q4.shape, weight.shape)
        self.assertLess(
            torch.mean((q8 - weight) ** 2).item(),
            torch.mean((q4 - weight) ** 2).item(),
        )
        self.assertGreaterEqual(q4.min().item(), -1.0)
        self.assertLessEqual(q4.max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
