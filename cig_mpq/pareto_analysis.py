from typing import Sequence

import numpy as np


def aggregate_strategy_rows(
    rows: Sequence[dict],
    baseline_name: str = "task_only",
    strategy_name: str = "calibration_constrained",
) -> dict:
    changed_rows = []
    all_deltas = []
    changed_deltas = []
    for row in rows:
        baseline = row["metrics"][baseline_name]
        strategy = row["metrics"][strategy_name]
        metric_names = sorted(set(baseline) & set(strategy))
        delta = {
            metric: float(strategy[metric]) - float(baseline[metric])
            for metric in metric_names
        }
        all_deltas.append(delta)
        configs = row.get("configs", {})
        changed = configs.get(baseline_name) != configs.get(strategy_name)
        if not changed:
            changed = any(abs(value) > 1e-15 for value in delta.values())
        if changed:
            changed_rows.append(row)
            changed_deltas.append(delta)

    metric_names = sorted(all_deltas[0]) if all_deltas else []
    improved_counts = {
        metric: sum(
            delta[metric] > 0.0
            if metric == "accuracy"
            else delta[metric] < 0.0
            for delta in changed_deltas
        )
        for metric in metric_names
    }

    def mean_delta(deltas: list[dict[str, float]]) -> dict[str, float]:
        if not deltas:
            return {metric: 0.0 for metric in metric_names}
        return {
            metric: float(np.mean([delta[metric] for delta in deltas]))
            for metric in metric_names
        }

    return {
        "baseline": baseline_name,
        "strategy": strategy_name,
        "total_cases": len(rows),
        "changed_cases": len(changed_rows),
        "unchanged_cases": len(rows) - len(changed_rows),
        "changed_improved_counts": improved_counts,
        "changed_mean_delta": mean_delta(changed_deltas),
        "all_mean_delta": mean_delta(all_deltas),
    }
