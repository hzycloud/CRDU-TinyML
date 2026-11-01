import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets

    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_accuracy = correct[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += mask.mean() * abs(bin_accuracy - bin_confidence)
    return float(ece)


def classification_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    return {
        "accuracy": float((predictions == targets).mean()),
        "ece": expected_calibration_error(probabilities, targets, n_bins=n_bins),
        "nll": float(-log_probabilities[np.arange(targets.shape[0]), targets].mean()),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def adaptive_calibration_error(probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE in equal-count bins (ties ordered stably)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    if len(targets) == 0 or n_bins < 1:
        raise ValueError("nonempty targets and positive bin count required")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    order = np.argsort(confidence, kind='stable')
    return float(sum(len(ids) / len(targets) * abs(correct[ids].mean() - confidence[ids].mean())
                     for ids in np.array_split(order, min(n_bins, len(targets)))))


def brier_decomposition(probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    """Classwise equal-width Murphy decomposition with explicit binning remainder.

    Sum over one-vs-rest classes matches the unnormalized multiclass Brier.
    Reliability/resolution are binned estimates; the remainder must be retained
    because individual forecasts vary inside each bin. This is descriptive.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if len(targets) == 0 or n_bins < 1:
        raise ValueError("nonempty targets and positive bin count required")
    y = np.eye(p.shape[1])[targets]
    prevalence = y.mean(axis=0)
    uncertainty = float(np.sum(prevalence * (1 - prevalence)))
    reliability = resolution = 0.
    for k in range(p.shape[1]):
        bin_ids = np.minimum((p[:, k] * n_bins).astype(int), n_bins - 1)
        for b in range(n_bins):
            mask = bin_ids == b
            if not mask.any():
                continue
            fraction = mask.mean()
            observed = y[mask, k].mean()
            reliability += fraction * (p[mask, k].mean() - observed)**2
            resolution += fraction * (observed - prevalence[k])**2
    score = float(np.mean(np.sum((p - y)**2, axis=1)))
    return dict(brier=score, reliability=float(reliability), resolution=float(resolution),
                uncertainty=uncertainty, binning_remainder=float(score - reliability + resolution - uncertainty))

