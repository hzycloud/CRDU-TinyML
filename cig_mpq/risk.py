import numpy as np

from .metrics import classification_metrics


def task_risk_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    return float(classification_metrics(logits, targets)["nll"])


def calibration_risk_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    temperature: float,
    brier_weight: float = 0.5,
) -> float:
    metrics = classification_metrics(
        np.asarray(logits, dtype=np.float64) / float(temperature),
        targets,
    )
    return float(metrics["nll"] + brier_weight * metrics["brier"])

