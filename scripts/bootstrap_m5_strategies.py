import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "ece", "nll", "brier", "weight_bytes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("comparison_rows", data.get("rows", []))


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    rows = load_rows(result_dir / "selection_comparison.json")
    rng = np.random.default_rng(args.seed)
    output = {}
    for strategy in ("calibration_only", "calibration_constrained"):
        differences = {metric: [] for metric in METRICS}
        for row in rows:
            baseline = row["metrics"].get("task_only")
            candidate = row["metrics"].get(strategy)
            if baseline is None or candidate is None:
                continue
            for metric in METRICS:
                differences[metric].append(
                    float(candidate[metric]) - float(baseline[metric])
                )
        summary = {}
        for metric, values in differences.items():
            values_array = np.asarray(values, dtype=np.float64)
            indices = rng.integers(
                0,
                len(values_array),
                size=(args.samples, len(values_array)),
            )
            bootstrap_means = values_array[indices].mean(axis=1)
            summary[metric] = {
                "n_pairs": int(len(values_array)),
                "mean_delta": float(values_array.mean()),
                "ci95_low": float(np.percentile(bootstrap_means, 2.5)),
                "ci95_high": float(np.percentile(bootstrap_means, 97.5)),
            }
        output[strategy] = summary
    output_path = result_dir / "bootstrap_summary.json"
    output_path.write_text(
        json.dumps(
            {
                "unit": "seed-budget pair",
                "samples": args.samples,
                "note": "Exploratory seed-budget bootstrap; not sample-level inference.",
                "results": output,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
