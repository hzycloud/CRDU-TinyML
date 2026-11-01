import numpy as np
from scipy.optimize import minimize_scalar

from .metrics import softmax


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return softmax(np.asarray(logits, dtype=np.float64) / temperature)


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probabilities = apply_temperature(logits, temperature)
        chosen = probabilities[np.arange(targets.shape[0]), targets]
        return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())

    result = minimize_scalar(
        objective,
        bounds=(-4.0, 4.0),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    return float(np.exp(result.x))

