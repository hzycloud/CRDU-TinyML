from itertools import combinations
from statistics import median
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .interaction import residualize_interactions


def _correlation_or_exact(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(spearmanr(first, second).statistic)


def summarize_residual_runs(
    runs: Mapping[str, tuple[Sequence[float], Sequence[float]]],
) -> dict:
    per_run = {}
    residuals = {}
    for name, (task_values, calibration_values) in runs.items():
        task = np.asarray(task_values, dtype=np.float64)
        calibration = np.asarray(calibration_values, dtype=np.float64)
        fit = residualize_interactions(task, calibration)
        calibration_variance = float(np.var(calibration))
        residual_variance = float(np.var(fit.residuals))
        residual_variance_ratio = (
            residual_variance / calibration_variance
            if calibration_variance > 0.0
            else 0.0
        )
        residuals[name] = fit.residuals
        per_run[name] = {
            "intercept": fit.intercept,
            "slope": fit.slope,
            "r_squared": fit.r_squared,
            "residual_variance_ratio": residual_variance_ratio,
            "residuals": fit.residuals.tolist(),
            "residual_signs": np.sign(fit.residuals).astype(int).tolist(),
        }

    comparisons = []
    for first, second in combinations(runs, 2):
        comparisons.append(
            {
                "first": first,
                "second": second,
                "residual_spearman": _correlation_or_exact(
                    residuals[first],
                    residuals[second],
                ),
                "residual_sign_agreement": float(
                    np.mean(np.sign(residuals[first]) == np.sign(residuals[second]))
                ),
            }
        )

    median_spearman = median(
        comparison["residual_spearman"] for comparison in comparisons
    )
    median_sign_agreement = median(
        comparison["residual_sign_agreement"] for comparison in comparisons
    )
    median_residual_variance_ratio = median(
        values["residual_variance_ratio"] for values in per_run.values()
    )
    passed = (
        median_residual_variance_ratio >= 0.05
        and median_spearman >= 0.5
        and median_sign_agreement >= 0.8
    )

    return {
        "per_run": per_run,
        "comparisons": comparisons,
        "independent_signal_gate": {
            "passed": bool(passed),
            "median_spearman": float(median_spearman),
            "median_sign_agreement": float(median_sign_agreement),
            "median_residual_variance_ratio": float(
                median_residual_variance_ratio
            ),
            "thresholds": {
                "median_spearman_min": 0.5,
                "median_sign_agreement_min": 0.8,
                "median_residual_variance_ratio_min": 0.05,
            },
        },
    }
