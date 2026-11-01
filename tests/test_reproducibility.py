import unittest

import torch

from cig_mpq.reproducibility import seed_everything


class ReproducibilityTest(unittest.TestCase):
    def test_seed_resets_random_stream_and_enables_deterministic_mode(self):
        seed_everything(7)
        first = torch.rand(8)
        seed_everything(7)
        second = torch.rand(8)

        torch.testing.assert_close(first, second)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)


if __name__ == "__main__":
    unittest.main()
