from typing import NamedTuple

import numpy as np


class InteractionResidualization(NamedTuple):
    intercept: float
    slope: float
    r_squared: float
    residuals: np.ndarray


def single_layer_delta(reference: float, layer_i: float) -> float:
    return float(layer_i - reference)


def pairwise_interaction(
    reference: float,
    layer_i: float,
    layer_j: float,
    pair: float,
) -> float:
    return float(pair - layer_i - layer_j + reference)


def residualize_interactions(
    task_interactions: np.ndarray,
    calibration_interactions: np.ndarray,
) -> InteractionResidualization:
    task = np.asarray(task_interactions, dtype=np.float64)
    calibration = np.asarray(calibration_interactions, dtype=np.float64)
    if task.shape != calibration.shape:
        raise ValueError("task and calibration interactions must have the same shape")

    design = np.column_stack([np.ones_like(task), task])
    intercept, slope = np.linalg.lstsq(design, calibration, rcond=None)[0]
    residuals = calibration - design @ np.array([intercept, slope])
    total_variation = float(np.sum((calibration - calibration.mean()) ** 2))
    residual_variation = float(np.sum(residuals**2))
    if total_variation == 0.0:
        r_squared = 1.0 if residual_variation == 0.0 else 0.0
    else:
        r_squared = 1.0 - residual_variation / total_variation
    return InteractionResidualization(
        intercept=float(intercept),
        slope=float(slope),
        r_squared=float(r_squared),
        residuals=residuals,
    )

