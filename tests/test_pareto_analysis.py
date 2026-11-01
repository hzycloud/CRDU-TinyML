import unittest

from cig_mpq.pareto_analysis import aggregate_strategy_rows


class ParetoAnalysisTest(unittest.TestCase):
    def test_aggregate_strategy_rows_counts_only_changed_cases(self):
        rows = [
            {
                "seed": 42,
                "budget": 1.0,
                "metrics": {
                    "task_only": {
                        "accuracy": 0.80,
                        "ece": 0.10,
                        "nll": 1.00,
                        "brier": 0.50,
                    },
                    "calibration_constrained": {
                        "accuracy": 0.79,
                        "ece": 0.08,
                        "nll": 1.01,
                        "brier": 0.51,
                    },
                },
            },
            {
                "seed": 43,
                "budget": 0.9,
                "metrics": {
                    "task_only": {
                        "accuracy": 0.70,
                        "ece": 0.09,
                        "nll": 1.10,
                        "brier": 0.55,
                    },
                    "calibration_constrained": {
                        "accuracy": 0.70,
                        "ece": 0.09,
                        "nll": 1.10,
                        "brier": 0.55,
                    },
                },
            },
        ]

        summary = aggregate_strategy_rows(rows)

        self.assertEqual(summary["total_cases"], 2)
        self.assertEqual(summary["changed_cases"], 1)
        self.assertEqual(summary["changed_improved_counts"]["ece"], 1)
        self.assertEqual(summary["changed_improved_counts"]["nll"], 0)
        self.assertAlmostEqual(
            summary["changed_mean_delta"]["ece"],
            -0.02,
        )


if __name__ == "__main__":
    unittest.main()
