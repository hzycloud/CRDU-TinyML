import unittest

import numpy as np

from cig_mpq.metrics import classification_metrics


class MetricsTest(unittest.TestCase):
    def test_extreme_wrong_prediction_uses_uncapped_log_loss(self):
        result = classification_metrics(np.array([[0., -100.]]), np.array([1]))
        self.assertAlmostEqual(result['nll'], 100.)

    def test_perfect_confidence_has_near_zero_error(self):
        logits = np.array([[20.0, -20.0], [-20.0, 20.0]], dtype=np.float64)
        targets = np.array([0, 1], dtype=np.int64)

        metrics = classification_metrics(logits, targets, n_bins=5)

        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertLess(metrics["ece"], 1e-6)
        self.assertLess(metrics["nll"], 1e-6)
        self.assertLess(metrics["brier"], 1e-6)

    def test_overconfident_wrong_prediction_has_positive_ece(self):
        logits = np.array([[10.0, -10.0], [10.0, -10.0]], dtype=np.float64)
        targets = np.array([1, 1], dtype=np.int64)

        metrics = classification_metrics(logits, targets, n_bins=5)

        self.assertAlmostEqual(metrics["accuracy"], 0.0)
        self.assertGreater(metrics["ece"], 0.9)
        self.assertGreater(metrics["nll"], 10.0)


if __name__ == "__main__":
    unittest.main()
