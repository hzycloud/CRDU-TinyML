import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_ONLY = ROOT / "results" / "cifar_pilot_det1"
ACTIVATION_INT8 = ROOT / "results" / "cifar_pilot_act8"


def main() -> None:
    runs = {
        "weight_only": WEIGHT_ONLY,
        "weight_plus_act8": ACTIVATION_INT8,
    }
    output = {}
    lines = [
        "# CIFAR-10 Weight-only vs INT8 Activation Pilot",
        "",
        "| Setting | FP32 Acc | W8 Task Risk | W8 Cal Risk | Interaction Spearman | Sign Agreement | Task Bits | Cal Bits |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for label, path in runs.items():
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        selections = json.loads((path / "selection_objectives.json").read_text(encoding="utf-8"))
        base = summary["configurations"][0]
        task_bits = ", ".join(selections["task_loss"]["bits"].keys()) or "none"
        cal_bits = ", ".join(selections["nll_brier"]["bits"].keys()) or "none"
        output[label] = {
            "fp32_metrics": summary["fp32_metrics"],
            "base_w8": {
                "task_risk": base["task_risk"],
                "calibration_risk": base["calibration_risk"],
                "metrics": base["metrics"],
            },
            "interaction_spearman": summary["interaction_spearman"],
            "sign_agreement": summary["interaction_sign_agreement"],
            "task_selection": selections["task_loss"],
            "nll_brier_selection": selections["nll_brier"],
            "brier_selection": selections["brier"],
            "ece_selection": selections["ece"],
        }
        lines.append(
            f"| {label} | {summary['fp32_metrics']['accuracy']:.4f} | "
            f"{base['task_risk']:.6f} | {base['calibration_risk']:.6f} | "
            f"{summary['interaction_spearman']:.3f} | {summary['interaction_sign_agreement']:.3f} | "
            f"{task_bits} | {cal_bits} |"
        )

    (ROOT / "results" / "cifar_activation_comparison.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (ROOT / "results" / "cifar_activation_comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
