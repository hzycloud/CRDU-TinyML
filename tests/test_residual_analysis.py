import unittest

import numpy as np

from cig_mpq.residual_analysis import summarize_residual_runs


class ResidualAnalysisTest(unittest.TestCase):
    def test_stable_residual_structure_is_detected_across_runs(self):
        task = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        independent_signal = np.array([1.0, -2.0, 0.0, 2.0, -1.0])
        runs = {
            name: (task, 0.5 * task + independent_signal)
            for name in ("seed42", "seed43", "seed44")
        }

        summary = summarize_residual_runs(runs)

        self.assertTrue(summary["independent_signal_gate"]["passed"])
        self.assertAlmostEqual(
            summary["independent_signal_gate"]["median_spearman"],
            1.0,
        )
        self.assertAlmostEqual(
            summary["independent_signal_gate"]["median_sign_agreement"],
            1.0,
        )

    def test_unstable_residual_structure_fails_gate(self):
        task = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        independent_signal = np.array([1.0, -2.0, 0.0, 2.0, -1.0])
        runs = {
            "seed42": (task, 0.5 * task + independent_signal),
            "seed43": (task, 0.5 * task - independent_signal),
            "seed44": (task, 0.5 * task + np.array([-1.0, 2.0, 0.0, -2.0, 1.0])),
        }

        summary = summarize_residual_runs(runs)

        self.assertFalse(summary["independent_signal_gate"]["passed"])

    def test_fully_explained_calibration_interaction_fails_gate(self):
        task = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        runs = {
            name: (task, 0.5 * task + 1.0)
            for name in ("seed42", "seed43", "seed44")
        }

        summary = summarize_residual_runs(runs)

        self.assertFalse(summary["independent_signal_gate"]["passed"])
        self.assertAlmostEqual(
            summary["independent_signal_gate"]["median_residual_variance_ratio"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
