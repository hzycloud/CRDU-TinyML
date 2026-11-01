import csv
import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.pareto_analysis import aggregate_strategy_rows


STRATEGIES = ("task_only", "calibration_only", "calibration_constrained")
METRICS = ("accuracy", "ece", "nll", "brier", "weight_bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "results" / "speech_pareto_128",
    )
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    selections = json.loads(
        (result_dir / "all_seed_selections.json").read_text(encoding="utf-8")
    )
    rows = []
    for seed, result in selections.items():
        for budget in result["budgets"]:
            configs = {}
            metrics = {}
            for strategy in STRATEGIES:
                selected = budget[strategy]
                if selected is None:
                    configs[strategy] = None
                    metrics[strategy] = None
                    continue
                configs[strategy] = selected["selection"]["config"]
                test = selected["test_calibrated_metrics"]
                metrics[strategy] = {
                    "accuracy": test["accuracy"],
                    "ece": test["ece"],
                    "nll": test["nll"],
                    "brier": test["brier"],
                    "weight_bytes": selected["selection"]["weight_bytes"],
                }
            rows.append(
                {
                    "seed": int(seed),
                    "budget": budget["budget"],
                    "configs": configs,
                    "metrics": metrics,
                }
            )

    complete = [
        row
        for row in rows
        if all(row["metrics"].get(strategy) is not None for strategy in STRATEGIES)
    ]
    calibration_summary = aggregate_strategy_rows(
        complete,
        baseline_name="task_only",
        strategy_name="calibration_only",
    )
    constrained_summary = aggregate_strategy_rows(
        complete,
        baseline_name="task_only",
        strategy_name="calibration_constrained",
    )
    output = {
        "calibration_only_vs_task_only": calibration_summary,
        "calibration_constrained_vs_task_only": constrained_summary,
        "rows": rows,
    }
    (result_dir / "selection_comparison.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (result_dir / "selection_comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "seed",
                "budget",
                "strategy",
                "config",
                "weight_bytes",
                "test_accuracy",
                "test_ece",
                "test_nll",
                "test_brier",
            ]
        )
        for row in rows:
            for strategy in STRATEGIES:
                metrics = row["metrics"].get(strategy)
                if metrics is None:
                    continue
                writer.writerow(
                    [
                        row["seed"],
                        row["budget"],
                        strategy,
                        row["configs"][strategy],
                        metrics["weight_bytes"],
                        metrics["accuracy"],
                        metrics["ece"],
                        metrics["nll"],
                        metrics["brier"],
                    ]
                )
    print(
        json.dumps(
            {
                "calibration_only_vs_task_only": calibration_summary,
                "calibration_constrained_vs_task_only": constrained_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
