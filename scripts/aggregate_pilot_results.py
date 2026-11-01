import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "seed42": ROOT / "results" / "cifar_pilot_det1",
    "seed43": ROOT / "results" / "cifar_pilot_seed43",
    "seed44": ROOT / "results" / "cifar_pilot_seed44",
}


def main() -> None:
    per_seed = {}
    for label, path in RUNS.items():
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        selections = json.loads((path / "selection_objectives.json").read_text(encoding="utf-8"))
        per_seed[label] = {
            "path": str(path),
            "fp32_accuracy": summary["fp32_metrics"]["accuracy"],
            "base_w8_task_risk": summary["configurations"][0]["task_risk"],
            "base_w8_calibration_risk": summary["configurations"][0]["calibration_risk"],
            "interaction_spearman": summary["interaction_spearman"],
            "sign_agreement": summary["interaction_sign_agreement"],
            "objectives": {
                name: {
                    "bits": item["bits"],
                    "accuracy": item["result"]["metrics"]["accuracy"],
                    "ece": item["result"]["metrics"]["ece"],
                    "nll": item["result"]["metrics"]["nll"],
                    "brier": item["result"]["metrics"]["brier"],
                    "temperature": item["result"]["temperature"],
                }
                for name, item in selections.items()
            },
        }

    metric_names = ["accuracy", "ece", "nll", "brier"]
    objective_names = ["task_loss", "nll_brier", "brier", "ece"]
    aggregate = {}
    for objective in objective_names:
        aggregate[objective] = {}
        for metric in metric_names:
            values = [
                seed["objectives"][objective][metric]
                for seed in per_seed.values()
            ]
            aggregate[objective][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "values": values,
            }

    output = {"per_seed": per_seed, "aggregate": aggregate}
    (ROOT / "results" / "cifar_pilot_aggregate.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# CIFAR-10 CIG-MPQ Pilot Aggregate",
        "",
        "| Seed | FP32 Acc | Interaction Spearman | Sign Agreement | Task Bits | NLL+Brier Bits | Brier Bits | ECE Bits |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for label, seed in per_seed.items():
        bits = {
            name: ", ".join(seed["objectives"][name]["bits"].keys()) or "none"
            for name in objective_names
        }
        lines.append(
            f"| {label} | {seed['fp32_accuracy']:.4f} | "
            f"{seed['interaction_spearman']:.3f} | {seed['sign_agreement']:.3f} | "
            f"{bits['task_loss']} | {bits['nll_brier']} | {bits['brier']} | {bits['ece']} |"
        )
    lines.extend(
        [
            "",
            "## Mean and standard deviation",
            "",
            "| Objective | Accuracy | ECE | NLL | Brier |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for objective in objective_names:
        row = [objective]
        for metric in metric_names:
            stats = aggregate[objective][metric]
            row.append(f"{stats['mean']:.4f} ± {stats['std']:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    (ROOT / "results" / "cifar_pilot_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
