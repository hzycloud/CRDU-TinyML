import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.residual_analysis import summarize_residual_runs


RUNS = {
    "seed42_weight_only": ROOT / "results" / "cifar_pilot_det1" / "interactions.csv",
    "seed43_weight_only": ROOT / "results" / "cifar_pilot_seed43" / "interactions.csv",
    "seed44_weight_only": ROOT / "results" / "cifar_pilot_seed44" / "interactions.csv",
    "seed42_int8_activation": ROOT / "results" / "cifar_pilot_act8" / "interactions.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_run(path: Path) -> tuple[list[str], list[float], list[float]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    pairs = [
        f"{row['layer_i']} -> {row['layer_j']}"
        for row in rows
    ]
    task = [float(row["task_interaction"]) for row in rows]
    calibration = [float(row["calibration_interaction"]) for row in rows]
    return pairs, task, calibration


def render_markdown(output: dict) -> str:
    primary = output["primary_seed_analysis"]
    gate = primary["independent_signal_gate"]
    lines = [
        "# CIFAR-10 校准交互残差分析",
        "",
        "本报告只使用已有真实交互结果，不新增或补写实验值。",
        "",
        "## 主要判断",
        "",
        f"- 跨种子独立信号门槛：{'通过' if gate['passed'] else '未通过'}",
        f"- 残差方差占比中位数：{gate['median_residual_variance_ratio']:.6f}",
        f"- 残差 Spearman 中位数：{gate['median_spearman']:.6f}",
        f"- 残差符号一致率中位数：{gate['median_sign_agreement']:.6f}",
        f"- 当前建议：`{output['recommendation']}`",
        "",
        "门槛为项目内探索性 Go/No-Go 规则（AI推断），不是领域标准。",
        "",
        "## 各种子解释度",
        "",
        "| 运行 | 截距 | 斜率 | R² | 残差方差占比 | 残差符号 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, values in primary["per_run"].items():
        signs = ", ".join(str(sign) for sign in values["residual_signs"])
        lines.append(
            f"| {name} | {values['intercept']:.9f} | {values['slope']:.6f} | "
            f"{values['r_squared']:.6f} | {values['residual_variance_ratio']:.6f} | "
            f"{signs} |"
        )

    lines.extend(
        [
            "",
            "## 层对残差",
            "",
            "| 层对 | seed42 | seed43 | seed44 |",
            "|---|---:|---:|---:|",
        ]
    )
    run_names = list(primary["per_run"])
    for index, pair in enumerate(output["pairs"]):
        values = [
            primary["per_run"][name]["residuals"][index]
            for name in run_names
        ]
        lines.append(
            f"| {pair} | " + " | ".join(f"{value:.9f}" for value in values) + " |"
        )

    sensitivity_gate = output["activation_sensitivity"]["independent_signal_gate"]
    lines.extend(
        [
            "",
            "## INT8 激活敏感性",
            "",
            f"- seed42 权重-only 与 INT8 激活的残差 Spearman："
            f"{sensitivity_gate['median_spearman']:.6f}",
            f"- 残差符号一致率：{sensitivity_gate['median_sign_agreement']:.6f}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    pairs_by_run = {}
    runs = {}
    provenance = {}
    for name, path in RUNS.items():
        pairs, task, calibration = load_run(path)
        pairs_by_run[name] = pairs
        runs[name] = (task, calibration)
        provenance[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }

    primary_names = [
        "seed42_weight_only",
        "seed43_weight_only",
        "seed44_weight_only",
    ]
    reference_pairs = pairs_by_run[primary_names[0]]
    for name in primary_names[1:]:
        if pairs_by_run[name] != reference_pairs:
            raise ValueError(f"layer pairs do not match for {name}")

    primary = summarize_residual_runs(
        {name: runs[name] for name in primary_names}
    )
    sensitivity = summarize_residual_runs(
        {
            "seed42_weight_only": runs["seed42_weight_only"],
            "seed42_int8_activation": runs["seed42_int8_activation"],
        }
    )
    median_r_squared = statistics.median(
        values["r_squared"] for values in primary["per_run"].values()
    )
    passed = primary["independent_signal_gate"]["passed"]
    output = {
        "input_provenance": provenance,
        "scope": {
            "dataset": "CIFAR-10 pilot",
            "primary_runs": primary_names,
            "layer_pairs": reference_pairs,
            "exploratory_only": True,
        },
        "pairs": reference_pairs,
        "median_r_squared": float(median_r_squared),
        "primary_seed_analysis": primary,
        "activation_sensitivity": sensitivity,
        "recommendation": (
            "KEEP_INTERACTION_MAIN_CLAIM_FOR_MORE_VALIDATION"
            if passed
            else "NARROW_TO_CALIBRATION_CONSTRAINED_MPQ"
        ),
        "threshold_note": (
            "Gate thresholds are project-specific exploratory Go/No-Go rules "
            "(AI inference), not field standards."
        ),
    }

    output_json = ROOT / "results" / "cifar_residual_interaction_analysis.json"
    output_md = ROOT / "results" / "cifar_residual_interaction_analysis.md"
    output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(output), encoding="utf-8")
    print(
        json.dumps(
            {
                "median_r_squared": output["median_r_squared"],
                "independent_signal_gate": primary["independent_signal_gate"],
                "activation_sensitivity": sensitivity["independent_signal_gate"],
                "recommendation": output["recommendation"],
                "json": str(output_json),
                "markdown": str(output_md),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
