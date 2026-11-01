import unittest

import numpy as np

from cig_mpq.bootstrap import paired_probability_bootstrap


class BootstrapTest(unittest.TestCase):
    def test_identical_probabilities_have_zero_paired_delta(self):
        probabilities = np.array(
            [
                [0.8, 0.2],
                [0.6, 0.4],
                [0.3, 0.7],
                [0.9, 0.1],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 1, 1, 0], dtype=np.int64)

        result = paired_probability_bootstrap(
            probabilities,
            probabilities,
            labels,
            n_bootstrap=100,
            seed=7,
        )

        for metric in ("accuracy", "ece", "nll", "brier"):
            self.assertAlmostEqual(result["mean_delta"][metric], 0.0)
            self.assertAlmostEqual(result["ci95_low"][metric], 0.0)
            self.assertAlmostEqual(result["ci95_high"][metric], 0.0)

    def test_candidate_with_higher_correct_probability_has_lower_nll(self):
        baseline = np.array(
            [[0.7, 0.3], [0.3, 0.7], [0.7, 0.3], [0.3, 0.7]],
            dtype=np.float64,
        )
        candidate = np.array(
            [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]],
            dtype=np.float64,
        )
        labels = np.array([0, 1, 0, 1], dtype=np.int64)

        result = paired_probability_bootstrap(
            baseline,
            candidate,
            labels,
            n_bootstrap=100,
            seed=9,
        )

        self.assertLess(result["mean_delta"]["nll"], 0.0)
        self.assertLess(result["mean_delta"]["brier"], 0.0)
        self.assertAlmostEqual(result["mean_delta"]["accuracy"], 0.0)

    def test_bootstrap_is_deterministic_for_fixed_seed(self):
        baseline = np.array([[0.6, 0.4], [0.2, 0.8], [0.7, 0.3]])
        candidate = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        labels = np.array([0, 1, 0])

        first = paired_probability_bootstrap(
            baseline,
            candidate,
            labels,
            n_bootstrap=50,
            seed=11,
        )
        second = paired_probability_bootstrap(
            baseline,
            candidate,
            labels,
            n_bootstrap=50,
            seed=11,
        )

        self.assertEqual(first["ci95_low"], second["ci95_low"])
        self.assertEqual(first["ci95_high"], second["ci95_high"])


if __name__ == "__main__":
    unittest.main()
