import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.bootstrap import METRICS, paired_probability_bootstrap


STRATEGIES = ("calibration_only", "calibration_constrained")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return data["probabilities"], data["labels"]


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    prediction_dir = result_dir / "predictions"
    index = json.loads((prediction_dir / "index.json").read_text(encoding="utf-8"))
    aggregate = {}
    cache = {}

    for strategy in STRATEGIES:
        case_results = []
        distributions = {metric: [] for metric in METRICS}
        for seed_text, seed_index in index.items():
            for budget_text, budget_index in seed_index["budgets"].items():
                baseline_mask = budget_index["task_only"]
                candidate_mask = budget_index.get(strategy)
                if baseline_mask is None or candidate_mask is None:
                    continue
                cache_key = (seed_text, int(baseline_mask), int(candidate_mask))
                if cache_key not in cache:
                    baseline_probs, labels = load_prediction(
                        prediction_dir / f"seed{seed_text}_mask{baseline_mask}.npz"
                    )
                    candidate_probs, candidate_labels = load_prediction(
                        prediction_dir / f"seed{seed_text}_mask{candidate_mask}.npz"
                    )
                    if not np.array_equal(labels, candidate_labels):
                        raise ValueError("prediction label order mismatch")
                    cache[cache_key] = paired_probability_bootstrap(
                        baseline_probs,
                        candidate_probs,
                        labels,
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed,
                    )
                result = cache[cache_key]
                case_results.append(
                    {
                        "seed": int(seed_text),
                        "budget": float(budget_text),
                        "baseline_mask": int(baseline_mask),
                        "candidate_mask": int(candidate_mask),
                        "same_config": int(baseline_mask) == int(candidate_mask),
                        "mean_delta": result["mean_delta"],
                        "ci95_low": result["ci95_low"],
                        "ci95_high": result["ci95_high"],
                    }
                )
                for metric in METRICS:
                    distributions[metric].append(
                        result["bootstrap_distributions"][metric]
                    )

        aggregate_distributions = {
            metric: np.mean(np.stack(values, axis=0), axis=0)
            for metric, values in distributions.items()
        }
        aggregate[strategy] = {
            "n_cases": len(case_results),
            "changed_cases": sum(not case["same_config"] for case in case_results),
            "mean_delta": {
                metric: float(
                    np.mean(
                        [case["mean_delta"][metric] for case in case_results]
                    )
                )
                for metric in METRICS
            },
            "ci95_low": {
                metric: float(np.percentile(values, 2.5))
                for metric, values in aggregate_distributions.items()
            },
            "ci95_high": {
                metric: float(np.percentile(values, 97.5))
                for metric, values in aggregate_distributions.items()
            },
            "cases": case_results,
        }

    output = {
        "task": args.task,
        "unit": "test sample, paired within seed-budget case",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "results": aggregate,
    }
    output_path = result_dir / "sample_level_bootstrap.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                strategy: {
                    "n_cases": values["n_cases"],
                    "changed_cases": values["changed_cases"],
                    "mean_delta": values["mean_delta"],
                    "ci95_low": values["ci95_low"],
                    "ci95_high": values["ci95_high"],
                }
                for strategy, values in aggregate.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
