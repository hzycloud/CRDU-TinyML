import csv
import hashlib
import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.pareto_analysis import aggregate_strategy_rows


STRATEGIES = ("task_only", "calibration_only", "calibration_constrained")
METRICS = ("accuracy", "ece", "nll", "brier")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "results" / "pareto_pilot",
    )
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    selections_path = result_dir / "all_seed_selections.json"
    selections = json.loads(selections_path.read_text(encoding="utf-8"))
    comparison_rows = []

    for seed, seed_result in selections.items():
        candidates_path = result_dir / f"seed{seed}_candidates.json"
        candidates = {
            row["config"]: row
            for row in json.loads(candidates_path.read_text(encoding="utf-8"))
        }
        for budget_result in seed_result["budgets"]:
            selected_configs = {}
            selected_metrics = {}
            for strategy in STRATEGIES:
                selected = budget_result[strategy]
                if selected is None:
                    selected_configs[strategy] = None
                    selected_metrics[strategy] = None
                    continue
                config = selected["config"]
                row = candidates[config]
                selected_configs[strategy] = config
                selected_metrics[strategy] = {
                    metric: row["eval_calibrated"][metric] for metric in METRICS
                }
                selected_metrics[strategy]["weight_bytes"] = row["weight_bytes"]
            comparison_rows.append(
                {
                    "seed": int(seed),
                    "budget": budget_result["budget"],
                    "configs": selected_configs,
                    "metrics": selected_metrics,
                }
            )

    finite_rows = [
        row
        for row in comparison_rows
        if all(row["metrics"].get(strategy) is not None for strategy in STRATEGIES)
    ]
    calibration_only_summary = aggregate_strategy_rows(
        finite_rows,
        baseline_name="task_only",
        strategy_name="calibration_only",
    )
    constrained_summary = aggregate_strategy_rows(
        finite_rows,
        baseline_name="task_only",
        strategy_name="calibration_constrained",
    )

    output = {
        "input_provenance": {
            "all_seed_selections": {
                "path": str(selections_path.relative_to(ROOT)),
                "sha256": sha256(selections_path),
            },
            "candidate_files": {
                seed: {
                    "path": str((result_dir / f"seed{seed}_candidates.json").relative_to(ROOT)),
                    "sha256": sha256(result_dir / f"seed{seed}_candidates.json"),
                }
                for seed in selections
            },
        },
        "evidence_boundary": (
            "CIFAR-10 pilot only; candidate selection uses search split, "
            "temperature is fit on a disjoint split, and all reported metrics "
            "are from the held-out test split."
        ),
        "comparison_rows": comparison_rows,
        "calibration_only_vs_task_only": calibration_only_summary,
        "calibration_constrained_vs_task_only": constrained_summary,
    }

    output_json = result_dir / "selection_comparison.json"
    output_json.write_text(
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
                "eval_accuracy",
                "eval_ece",
                "eval_nll",
                "eval_brier",
            ]
        )
        for row in comparison_rows:
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

    lines = [
        "# 校准约束 Pareto 试点汇总",
        "",
        "只汇总 `results/pareto_pilot/` 中已执行的三 seed × 64 配置结果。",
        "",
        "## 任务损失选择 vs 校准约束选择",
        "",
        f"- 总 seed-预算组合：{constrained_summary['total_cases']}",
        f"- 配置发生改变：{constrained_summary['changed_cases']}",
        f"- 改变案例中 ECE 改善："
        f"{constrained_summary['changed_improved_counts']['ece']}/"
        f"{constrained_summary['changed_cases']}",
        f"- 改变案例中 NLL 改善："
        f"{constrained_summary['changed_improved_counts']['nll']}/"
        f"{constrained_summary['changed_cases']}",
        f"- 改变案例中 Accuracy 改善："
        f"{constrained_summary['changed_improved_counts']['accuracy']}/"
        f"{constrained_summary['changed_cases']}",
        "",
        "改变案例的测试集平均差值（约束减任务）：",
        "",
        "| 指标 | 平均差值 |",
        "|---|---:|",
    ]
    for metric in METRICS:
        lines.append(
            f"| {metric} | "
            f"{constrained_summary['changed_mean_delta'][metric]:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Calibration-only 对照",
            "",
            f"- 与 Task-only 选择相同的组合："
            f"{calibration_only_summary['unchanged_cases']}/"
            f"{calibration_only_summary['total_cases']}",
            "",
            "## 判定",
            "",
            (
                "校准约束能稳定降低 ECE，但当前证据不支持它在 Accuracy、"
                "NLL 和 Brier 上形成一致优势。因此应停止把 CIG-MPQ 写成"
                "已证明的主方法，转为校准目标权衡研究或负结果诊断路线。"
            ),
            "",
        ]
    )
    output_md = result_dir / "selection_comparison.md"
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "calibration_only_vs_task_only": calibration_only_summary,
                "calibration_constrained_vs_task_only": constrained_summary,
                "json": str(output_json),
                "csv": str(result_dir / "selection_comparison.csv"),
                "markdown": str(output_md),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
