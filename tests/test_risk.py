import unittest

import numpy as np

from cig_mpq.risk import calibration_risk_from_logits, task_risk_from_logits


class RiskTest(unittest.TestCase):
    def test_task_risk_is_cross_entropy(self):
        logits = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        targets = np.array([0, 0], dtype=np.int64)

        risk = task_risk_from_logits(logits, targets)

        expected = (np.log1p(np.exp(-2.0)) + np.log1p(np.exp(2.0))) / 2.0
        self.assertAlmostEqual(risk, float(expected), places=6)

    def test_calibration_risk_uses_temperature_and_brier(self):
        logits = np.array([[4.0, -4.0], [-4.0, 4.0]], dtype=np.float64)
        targets = np.array([0, 1], dtype=np.int64)

        risk = calibration_risk_from_logits(logits, targets, temperature=2.0, brier_weight=0.5)

        self.assertGreaterEqual(risk, 0.0)
        self.assertLess(risk, 1.0)


if __name__ == "__main__":
    unittest.main()
