import unittest

import numpy as np

from cig_mpq.search import solve_bitwidth_assignment


class SearchTest(unittest.TestCase):
    def test_budget_forces_one_w4_layer(self):
        solution = solve_bitwidth_assignment(
            deltas=np.array([0.1, 0.3]),
            interactions={(0, 1): 0.0},
            flash_w8=np.array([4.0, 4.0]),
            flash_budget=6.0,
        )

        np.testing.assert_array_equal(solution, np.array([1, 0]))

    def test_negative_interaction_can_favor_joint_quantization(self):
        solution = solve_bitwidth_assignment(
            deltas=np.array([0.4, 0.4]),
            interactions={(0, 1): -0.5},
            flash_w8=np.array([4.0, 4.0]),
            flash_budget=4.0,
        )

        np.testing.assert_array_equal(solution, np.array([1, 1]))


if __name__ == "__main__":
    unittest.main()
