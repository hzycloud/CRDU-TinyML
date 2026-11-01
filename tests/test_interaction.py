import unittest

import numpy as np

from cig_mpq.interaction import (
    pairwise_interaction,
    residualize_interactions,
    single_layer_delta,
)


class InteractionTest(unittest.TestCase):
    def test_additive_risk_produces_zero_interaction(self):
        interaction = pairwise_interaction(
            reference=1.0,
            layer_i=1.2,
            layer_j=1.4,
            pair=1.6,
        )

        self.assertAlmostEqual(interaction, 0.0)

    def test_positive_non_additive_risk_is_detected(self):
        interaction = pairwise_interaction(
            reference=1.0,
            layer_i=1.2,
            layer_j=1.4,
            pair=1.8,
        )

        self.assertAlmostEqual(interaction, 0.2)
        self.assertAlmostEqual(single_layer_delta(1.0, 1.2), 0.2)

    def test_calibration_interaction_fully_explained_by_task_loss_has_zero_residual(self):
        task = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        calibration = 2.5 * task + 0.75

        result = residualize_interactions(task, calibration)

        np.testing.assert_allclose(result.residuals, 0.0, atol=1e-12)
        self.assertAlmostEqual(result.slope, 2.5)
        self.assertAlmostEqual(result.intercept, 0.75)
        self.assertAlmostEqual(result.r_squared, 1.0)

    def test_orthogonal_independent_signal_is_preserved(self):
        task = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        independent_signal = np.array([1.0, -2.0, 0.0, 2.0, -1.0])
        calibration = 0.5 * task + independent_signal

        result = residualize_interactions(task, calibration)

        np.testing.assert_allclose(result.residuals, independent_signal, atol=1e-12)
        self.assertAlmostEqual(result.slope, 0.5)
        self.assertAlmostEqual(result.intercept, 0.0)


if __name__ == "__main__":
    unittest.main()
