import numpy as np


METRICS = ("accuracy", "ece", "nll", "brier")


def metrics_from_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or labels.shape != (probabilities.shape[0],):
        raise ValueError("invalid probability or label shape")
    chosen = probabilities[np.arange(labels.shape[0]), labels]
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    confidence = probabilities.max(axis=1)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]

    bin_indices = np.minimum(
        (confidence * n_bins).astype(np.int64),
        n_bins - 1,
    )
    counts = np.bincount(bin_indices, minlength=n_bins)
    correct_sums = np.bincount(
        bin_indices,
        weights=correct.astype(np.float64),
        minlength=n_bins,
    )
    confidence_sums = np.bincount(
        bin_indices,
        weights=confidence,
        minlength=n_bins,
    )
    ece = float(np.abs(correct_sums - confidence_sums).sum() / labels.shape[0])
    return {
        "accuracy": float(correct.mean()),
        "ece": ece,
        "nll": float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean()),
        "brier": float(np.sum((probabilities - one_hot) ** 2, axis=1).mean()),
    }


def paired_probability_bootstrap(
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    n_bins: int = 15,
) -> dict:
    baseline = np.asarray(baseline_probabilities, dtype=np.float64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate probability shapes differ")
    if baseline.ndim != 2 or labels.shape != (baseline.shape[0],):
        raise ValueError("invalid probability or label shape")

    baseline_metrics = metrics_from_probabilities(baseline, labels, n_bins=n_bins)
    candidate_metrics = metrics_from_probabilities(candidate, labels, n_bins=n_bins)
    mean_delta = {
        metric: candidate_metrics[metric] - baseline_metrics[metric]
        for metric in METRICS
    }

    rng = np.random.default_rng(seed)
    distributions = {metric: np.empty(n_bootstrap, dtype=np.float64) for metric in METRICS}
    for bootstrap_index in range(n_bootstrap):
        indices = rng.integers(0, labels.shape[0], size=labels.shape[0])
        sampled_baseline = metrics_from_probabilities(
            baseline[indices],
            labels[indices],
            n_bins=n_bins,
        )
        sampled_candidate = metrics_from_probabilities(
            candidate[indices],
            labels[indices],
            n_bins=n_bins,
        )
        for metric in METRICS:
            distributions[metric][bootstrap_index] = (
                sampled_candidate[metric] - sampled_baseline[metric]
            )

    return {
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "mean_delta": mean_delta,
        "ci95_low": {
            metric: float(np.percentile(values, 2.5))
            for metric, values in distributions.items()
        },
        "ci95_high": {
            metric: float(np.percentile(values, 97.5))
            for metric, values in distributions.items()
        },
        "bootstrap_distributions": distributions,
    }
