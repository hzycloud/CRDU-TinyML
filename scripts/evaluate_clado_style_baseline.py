import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.search import solve_bitwidth_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[1.0, 0.95, 0.90, 0.85, 0.80],
    )
    return parser.parse_args()


def bitmask_from_selected(selected: np.ndarray) -> int:
    return int(sum(int(flag) << index for index, flag in enumerate(selected)))


def task_risk(row: dict) -> float:
    if "search_raw_nll" in row:
        return float(row["search_raw_nll"])
    return float(row["search_raw"]["nll"])


def test_raw(row: dict) -> dict:
    return row.get("test_raw_metrics", row.get("eval_raw"))


def test_calibrated(row: dict) -> dict:
    return row.get("test_calibrated_metrics", row.get("eval_calibrated"))


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    outputs = []
    for candidates_path in sorted(result_dir.glob("seed*_candidates.json")):
        rows = json.loads(candidates_path.read_text(encoding="utf-8"))
        rows_by_mask = {int(row["mask"]): row for row in rows}
        layer_names = rows[0]["layer_names"]
        n_layers = len(layer_names)
        base = rows_by_mask[0]
        deltas = np.asarray(
            [
                task_risk(rows_by_mask[1 << index]) - task_risk(base)
                for index in range(n_layers)
            ],
            dtype=np.float64,
        )
        interactions = {}
        for first in range(n_layers):
            for second in range(first + 1, n_layers):
                mask = (1 << first) | (1 << second)
                interactions[(first, second)] = (
                    task_risk(rows_by_mask[mask])
                    - task_risk(rows_by_mask[1 << first])
                    - task_risk(rows_by_mask[1 << second])
                    + task_risk(base)
                )
        flash_w8 = np.asarray(
            [
                2.0
                * (
                    base["weight_bytes"]
                    - rows_by_mask[1 << index]["weight_bytes"]
                )
                for index in range(n_layers)
            ],
            dtype=np.float64,
        )
        budget_rows = []
        for budget_fraction in args.budgets:
            selected = solve_bitwidth_assignment(
                deltas,
                interactions,
                flash_w8,
                float(flash_w8.sum() * budget_fraction),
            )
            mask = bitmask_from_selected(selected)
            row = rows_by_mask[mask]
            budget_rows.append(
                {
                    "budget": budget_fraction,
                    "selected": selected.tolist(),
                    "mask": mask,
                    "bits": row["bits"],
                    "weight_bytes": row["weight_bytes"],
                    "flash_fraction": row["flash_fraction"],
                    "temperature": row["temperature"],
                    "search_raw_nll": task_risk(row),
                    "search_calibrated_metrics": row["search_calibrated"],
                    "test_raw_metrics": test_raw(row),
                    "test_calibrated_metrics": test_calibrated(row),
                }
            )
        seed = int(candidates_path.stem.split("_")[0].removeprefix("seed"))
        outputs.append(
            {
                "seed": seed,
                "layer_names": layer_names,
                "deltas": deltas.tolist(),
                "pair_count": len(interactions),
                "flash_w8": flash_w8.tolist(),
                "budgets": budget_rows,
            }
        )

    payload = {
        "note": (
            "Adapted CLADO-style baseline using task-loss single-layer and "
            "pairwise differences with the same integer search and budgets. "
            "This is not a reproduction of the original CLADO code."
        ),
        "results": outputs,
    }
    (result_dir / "clado_style_baseline.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (result_dir / "clado_style_baseline.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "seed",
                "budget",
                "mask",
                "weight_bytes",
                "flash_fraction",
                "test_accuracy",
                "test_ece",
                "test_nll",
                "test_brier",
            ]
        )
        for result in outputs:
            for row in result["budgets"]:
                test = row["test_calibrated_metrics"]
                writer.writerow(
                    [
                        result["seed"],
                        row["budget"],
                        row["mask"],
                        row["weight_bytes"],
                        row["flash_fraction"],
                        test["accuracy"],
                        test["ece"],
                        test["nll"],
                        test["brier"],
                    ]
                )
    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
