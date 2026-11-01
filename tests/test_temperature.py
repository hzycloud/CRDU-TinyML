import unittest

import numpy as np

from cig_mpq.metrics import classification_metrics
from cig_mpq.temperature import apply_temperature, fit_temperature


class TemperatureTest(unittest.TestCase):
    def test_temperature_reduces_overconfident_nll(self):
        logits = np.array(
            [
                [6.0, -6.0],
                [6.0, -6.0],
                [-6.0, 6.0],
                [-6.0, 6.0],
            ],
            dtype=np.float64,
        )
        targets = np.array([0, 1, 0, 1], dtype=np.int64)

        temperature = fit_temperature(logits, targets)
        calibrated = apply_temperature(logits, temperature)
        before = classification_metrics(logits, targets, n_bins=4)
        after = classification_metrics(np.log(calibrated), targets, n_bins=4)

        self.assertGreater(temperature, 1.0)
        self.assertLess(after["nll"], before["nll"])


if __name__ == "__main__":
    unittest.main()
