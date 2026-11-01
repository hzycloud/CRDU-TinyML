from typing import Mapping, Sequence


def add_calibration_risk(
    metrics: Mapping[str, float],
    brier_weight: float = 0.5,
) -> dict[str, float]:
    enriched = {key: float(value) for key, value in metrics.items()}
    enriched["calibration_risk"] = (
        float(metrics["nll"]) + brier_weight * float(metrics["brier"])
    )
    return enriched


def select_calibration_constrained(
    candidates: Sequence[Mapping[str, float]],
    max_ece: float,
    min_accuracy: float,
    task_risk_key: str = "task_risk",
) -> Mapping[str, float]:
    feasible = [
        candidate
        for candidate in candidates
        if float(candidate["ece"]) <= max_ece
        and float(candidate["accuracy"]) >= min_accuracy
    ]
    if not feasible:
        raise ValueError("no candidate satisfies the calibration constraints")
    return min(feasible, key=lambda candidate: float(candidate[task_risk_key]))


def select_for_budget(
    candidates: Sequence[Mapping[str, float]],
    budget: float,
    baseline_view: Mapping[str, float],
    accuracy_tolerance: float = 0.01,
) -> dict:
    within_budget = [
        candidate
        for candidate in candidates
        if float(candidate["flash_fraction"]) <= budget + 1e-12
    ]
    accuracy_floor = float(baseline_view["accuracy"]) - accuracy_tolerance
    ece_cap = float(baseline_view["ece"])
    eligible = [
        candidate
        for candidate in within_budget
        if float(candidate["accuracy"]) >= accuracy_floor
    ]
    if not eligible:
        return {
            "budget": budget,
            "accuracy_floor": accuracy_floor,
            "ece_cap": ece_cap,
            "task_only": None,
            "calibration_only": None,
            "calibration_constrained": None,
        }

    task_only = min(eligible, key=lambda candidate: float(candidate["task_risk"]))
    calibration_only = min(
        eligible,
        key=lambda candidate: float(candidate["calibration_risk"]),
    )
    try:
        constrained = select_calibration_constrained(
            eligible,
            max_ece=ece_cap,
            min_accuracy=accuracy_floor,
        )
    except ValueError:
        constrained = None
    return {
        "budget": budget,
        "accuracy_floor": accuracy_floor,
        "ece_cap": ece_cap,
        "task_only": task_only,
        "calibration_only": calibration_only,
        "calibration_constrained": constrained,
    }


def pareto_frontier(
    candidates: Sequence[Mapping[str, float]],
    objectives: Sequence[str],
) -> list[Mapping[str, float]]:
    frontier = []
    for candidate in candidates:
        values = [float(candidate[objective]) for objective in objectives]
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            other_values = [float(other[objective]) for objective in objectives]
            if all(
                other_value <= value
                for other_value, value in zip(other_values, values)
            ) and any(
                other_value < value
                for other_value, value in zip(other_values, values)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier
