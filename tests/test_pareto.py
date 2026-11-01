import unittest

from cig_mpq.pareto import (
    add_calibration_risk,
    pareto_frontier,
    select_for_budget,
    select_calibration_constrained,
)


class ParetoTest(unittest.TestCase):
    def test_calibration_constraint_filters_candidates_before_task_selection(self):
        candidates = [
            {
                "config": "task_best_but_miscalibrated",
                "task_risk": 1.0,
                "ece": 0.20,
                "accuracy": 0.80,
            },
            {
                "config": "calibration_feasible",
                "task_risk": 1.1,
                "ece": 0.08,
                "accuracy": 0.79,
            },
            {
                "config": "too_inaccurate",
                "task_risk": 1.2,
                "ece": 0.05,
                "accuracy": 0.70,
            },
        ]

        selected = select_calibration_constrained(
            candidates,
            max_ece=0.10,
            min_accuracy=0.75,
        )

        self.assertEqual(selected["config"], "calibration_feasible")

    def test_infeasible_calibration_constraint_raises(self):
        with self.assertRaises(ValueError):
            select_calibration_constrained(
                [{"task_risk": 1.0, "ece": 0.20, "accuracy": 0.80}],
                max_ece=0.10,
                min_accuracy=0.75,
            )

    def test_pareto_frontier_removes_dominated_candidates(self):
        candidates = [
            {"config": "small_fast", "flash": 5.0, "risk": 1.2},
            {"config": "dominated", "flash": 6.0, "risk": 1.3},
            {"config": "accurate_large", "flash": 8.0, "risk": 1.0},
            {"config": "tradeoff", "flash": 7.0, "risk": 1.1},
        ]

        frontier = pareto_frontier(candidates, objectives=("flash", "risk"))

        self.assertEqual(
            {candidate["config"] for candidate in frontier},
            {"small_fast", "accurate_large", "tradeoff"},
        )

    def test_calibration_risk_is_derived_from_nll_and_brier(self):
        metrics = add_calibration_risk(
            {
                "accuracy": 0.8,
                "ece": 0.05,
                "nll": 1.2,
                "brier": 0.4,
            }
        )

        self.assertAlmostEqual(metrics["calibration_risk"], 1.4)

    def test_budget_selection_uses_flat_candidate_views(self):
        candidates = [
            {
                "config": "too_large",
                "flash_fraction": 1.0,
                "accuracy": 0.80,
                "ece": 0.04,
                "task_risk": 0.9,
                "calibration_risk": 1.1,
            },
            {
                "config": "within_budget",
                "flash_fraction": 0.90,
                "accuracy": 0.79,
                "ece": 0.05,
                "task_risk": 1.0,
                "calibration_risk": 1.2,
            },
        ]
        baseline = {
            "accuracy": 0.80,
            "ece": 0.06,
        }

        selection = select_for_budget(
            candidates,
            budget=0.95,
            baseline_view=baseline,
        )

        self.assertEqual(selection["task_only"]["config"], "within_budget")
        self.assertEqual(
            selection["calibration_constrained"]["config"],
            "within_budget",
        )


if __name__ == "__main__":
    unittest.main()
