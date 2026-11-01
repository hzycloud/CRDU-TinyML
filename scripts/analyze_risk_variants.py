import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.experiment import risk_delta_and_interactions


def matrix_from_interactions(
    layer_names: list[str],
    deltas: list[float],
    interactions: dict[tuple[int, int], float],
) -> np.ndarray:
    matrix = np.full((len(layer_names), len(layer_names)), np.nan)
    np.fill_diagonal(matrix, deltas)
    for (i, j), value in interactions.items():
        matrix[i, j] = matrix[j, i] = value
    return matrix


def main() -> None:
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "cifar_pilot_det1"
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    layer_names = summary["candidate_layers"]
    pairs = [tuple(pair) for pair in summary["pairs"]]
    metrics = list(csv.DictReader((result_dir / "config_metrics.csv").open(encoding="utf-8")))
    metrics_by_config = {
        row["config"]: {
            key: float(value)
            for key, value in row.items()
            if key != "config"
        }
        for row in metrics
    }

    variants = {
        "task_loss": "task_risk",
        "nll_brier": "calibration_risk",
        "brier": "brier",
        "ece": "ece",
    }
    vectors = {}
    variant_summary = {}
    for label, risk_key in variants.items():
        deltas, interactions = risk_delta_and_interactions(
            metrics_by_config,
            layer_names,
            pairs,
            risk_key,
        )
        vectors[label] = np.array(list(interactions.values()))
        variant_summary[label] = {
            "deltas": deltas,
            "interactions": [
                {"pair": list(pair), "value": interactions[(i, j)]}
                for i, j in sorted(interactions)
                for pair in [[layer_names[i], layer_names[j]]]
            ],
        }

    labels = list(variants)
    correlation = np.eye(len(labels))
    for i, first in enumerate(labels):
        for j, second in enumerate(labels):
            if i == j:
                continue
            if np.std(vectors[first]) > 0 and np.std(vectors[second]) > 0:
                correlation[i, j] = spearmanr(vectors[first], vectors[second]).statistic
            else:
                correlation[i, j] = np.nan

    sign_agreement = {
        label: float(np.mean(np.sign(vectors["task_loss"]) == np.sign(vectors[label])))
        for label in labels
    }

    fig, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Spearman correlation of interaction vectors")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = correlation[i, j]
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axis, shrink=0.85)
    fig.savefig(result_dir / "risk_variant_correlations.png", dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    axis.scatter(vectors["task_loss"], vectors["brier"], color="#1F4E79", s=70)
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axvline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlabel("Task-loss interaction")
    axis.set_ylabel("Brier interaction")
    axis.set_title("Task-loss vs Brier interaction")
    fig.savefig(result_dir / "task_vs_brier_interactions.png", dpi=300)
    plt.close(fig)

    output = {
        "variants": variant_summary,
        "spearman_correlation": {
            first: {second: float(correlation[i, j]) for j, second in enumerate(labels)}
            for i, first in enumerate(labels)
        },
        "sign_agreement_with_task_loss": sign_agreement,
    }
    (result_dir / "risk_variant_summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(output["spearman_correlation"], indent=2))


if __name__ == "__main__":
    main()
