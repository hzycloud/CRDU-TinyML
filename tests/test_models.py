import unittest

import torch

from cig_mpq.models import AudioDSCNN, DSCNN


class ModelTest(unittest.TestCase):
    def test_dscnn_output_shape(self):
        model = DSCNN(num_classes=12, in_channels=1)
        inputs = torch.randn(2, 1, 40, 101)
        outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (2, 12))

    def test_dscnn_supports_rgb_input(self):
        model = DSCNN(num_classes=10, in_channels=3)
        inputs = torch.randn(2, 3, 32, 32)
        outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (2, 10))

    def test_audio_dscnn_accepts_49_by_10_mfcc_input(self):
        model = AudioDSCNN(num_classes=12)
        inputs = torch.randn(2, 1, 49, 10)
        outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (2, 12))


if __name__ == "__main__":
    unittest.main()
