import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.data import build_imagefolder
from cig_mpq.experiment import build_quantization_configs, choose_candidate_layers
from cig_mpq.interaction import pairwise_interaction
from cig_mpq.metrics import classification_metrics
from cig_mpq.models import DSCNN
from cig_mpq.quantization import install_activation_quantization_hooks, quantize_mixed_precision
from cig_mpq.reproducibility import seed_everything
from cig_mpq.search import solve_bitwidth_assignment
from cig_mpq.temperature import fit_temperature
from cig_mpq.training import evaluate_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, default=ROOT / "datasets" / "cifar10" / "extracted" / "cifar10" / "train")
    parser.add_argument("--test-root", type=Path, default=ROOT / "datasets" / "cifar10" / "extracted" / "cifar10" / "test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "cifar_pilot")
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--temp-limit", type=int, default=2000)
    parser.add_argument("--eval-limit", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activation-bits", type=int, default=8)
    return parser.parse_args()


def split_dataset(dataset, train_limit: int, temp_limit: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    needed = train_limit + temp_limit
    if needed > len(indices):
        raise ValueError(f"requested split {needed} exceeds dataset size {len(indices)}")
    train_indices = indices[:train_limit]
    temp_indices = indices[train_limit:needed]
    return Subset(dataset, train_indices), Subset(dataset, temp_indices)


def train_model(model, loader, device, epochs, learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, targets in tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.shape[0]
            seen += targets.shape[0]
        history.append({"epoch": epoch, "train_loss": running_loss / seen})
    return history


def evaluate_configuration(model, config, temp_loader, eval_loader, device, activation_bits=8):
    candidate = copy.deepcopy(model)
    quantize_mixed_precision(candidate, config, default_bits=8)
    handles = install_activation_quantization_hooks(candidate, bits=activation_bits)
    try:
        _, temp_logits, temp_labels = evaluate_logits(candidate, temp_loader, device)
        metrics, eval_logits, eval_labels = evaluate_logits(candidate, eval_loader, device)
    finally:
        for handle in handles:
            handle.remove()
    temperature = fit_temperature(temp_logits, temp_labels)
    calibrated_metrics = classification_metrics(eval_logits / temperature, eval_labels)
    task_risk = float(metrics["nll"])
    calibration_risk = float(calibrated_metrics["nll"] + 0.5 * calibrated_metrics["brier"])
    return {
        "temperature": float(temperature),
        "task_risk": task_risk,
        "calibration_risk": calibration_risk,
        "metrics": calibrated_metrics,
        "raw_metrics": metrics,
    }


def evaluate_selected_layers(
    model,
    selected: np.ndarray,
    layer_names: list[str],
    loader,
    device,
    activation_bits=8,
):
    bits = {
        name: 4
        for name, flag in zip(layer_names, selected.tolist())
        if flag == 1
    }
    candidate = copy.deepcopy(model)
    quantize_mixed_precision(candidate, bits, default_bits=8)
    handles = install_activation_quantization_hooks(candidate, bits=activation_bits)
    try:
        metrics, logits, labels = evaluate_logits(candidate, loader, device)
    finally:
        for handle in handles:
            handle.remove()
    return metrics, bits, logits, labels


def write_config_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "config",
        "temperature",
        "task_risk",
        "calibration_risk",
        "accuracy",
        "ece",
        "nll",
        "brier",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "config": row["config"],
                    "temperature": row["temperature"],
                    "task_risk": row["task_risk"],
                    "calibration_risk": row["calibration_risk"],
                    "accuracy": row["metrics"]["accuracy"],
                    "ece": row["metrics"]["ece"],
                    "nll": row["metrics"]["nll"],
                    "brier": row["metrics"]["brier"],
                }
            )


def short_layer_label(name: str) -> str:
    mapping = {
        "features.0": "Conv1",
        "features.4.depthwise": "DW1",
        "features.4.pointwise": "PW1",
        "features.6.depthwise": "DW2",
        "features.6.pointwise": "PW2",
        "features.8.depthwise": "DW3",
    }
    return mapping.get(name, name[-8:])


def short_config_label(config: str) -> str:
    if config == "base":
        return "Base W8"
    body = config.removeprefix("w4:")
    if "+" not in body:
        return f"W4:{short_layer_label(body)}"
    first, second = body.split("+", 1)
    return f"{short_layer_label(first)}+{short_layer_label(second)}"


def plot_interaction_matrices(path: Path, names: list[str], task_matrix: np.ndarray, cal_matrix: np.ndarray) -> None:
    short_names = [short_layer_label(name) for name in names]
    task_off_diagonal = task_matrix.copy()
    cal_off_diagonal = cal_matrix.copy()
    np.fill_diagonal(task_off_diagonal, np.nan)
    np.fill_diagonal(cal_off_diagonal, np.nan)
    fig = plt.figure(figsize=(11.5, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.06])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    color_axis = fig.add_subplot(grid[0, 2])
    image = None
    for axis, matrix, title in (
        (axes[0], task_off_diagonal, "Task-loss interaction"),
        (axes[1], cal_off_diagonal, "Calibrated-risk interaction"),
    ):
        image = axis.imshow(np.ma.masked_invalid(matrix), cmap="RdBu_r")
        axis.set_title(title)
        axis.set_xticks(range(len(names)), short_names, rotation=45, ha="right")
        axis.set_yticks(range(len(names)), short_names)
        axis.set_aspect("equal")
    fig.colorbar(image, cax=color_axis, label="Pairwise interaction")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_config_risks(path: Path, rows: list[dict]) -> None:
    labels = [short_config_label(row["config"]) for row in rows]
    task = [row["task_risk"] for row in rows]
    calibrated = [row["calibration_risk"] for row in rows]
    x = np.arange(len(rows))
    width = 0.38
    fig, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    axis.bar(x - width / 2, task, width, label="Task loss")
    axis.bar(x + width / 2, calibrated, width, label="Calibrated risk")
    axis.set_ylabel("Risk")
    axis.set_xticks(x, labels, rotation=55, ha="right")
    axis.legend()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_train = build_imagefolder(args.train_root, image_size=32, normalize=True)
    test_dataset = build_imagefolder(args.test_root, image_size=32, normalize=True)
    train_subset, temp_subset = split_dataset(full_train, args.train_limit, args.temp_limit, args.seed)
    eval_indices = list(range(min(args.eval_limit, len(test_dataset))))
    eval_subset = Subset(test_dataset, eval_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    temp_loader = DataLoader(temp_subset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = DSCNN(num_classes=len(full_train.classes), in_channels=3).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.time()
    history = train_model(model, train_loader, device, args.epochs, args.learning_rate)
    training_seconds = time.time() - started

    fp32_metrics, _, _ = evaluate_logits(model, eval_loader, device)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": full_train.classes,
            "args": vars(args),
        },
        args.output_dir / "model.pt",
    )

    layer_names = choose_candidate_layers(model, max_layers=6)
    pairs = list(zip(layer_names[:-1], layer_names[1:]))
    configs = build_quantization_configs(layer_names, pairs)

    rows = []
    started = time.time()
    for config_name, bits in tqdm(configs.items(), desc="evaluating configurations"):
        result = evaluate_configuration(
            model,
            bits,
            temp_loader,
            eval_loader,
            device,
            activation_bits=args.activation_bits,
        )
        result["config"] = config_name
        rows.append(result)
    evaluation_seconds = time.time() - started

    by_config = {row["config"]: row for row in rows}
    task_matrix = np.full((len(layer_names), len(layer_names)), np.nan)
    cal_matrix = np.full_like(task_matrix, np.nan)
    base_task = by_config["base"]["task_risk"]
    base_cal = by_config["base"]["calibration_risk"]

    for index, name in enumerate(layer_names):
        task_matrix[index, index] = by_config[f"w4:{name}"]["task_risk"] - base_task
        cal_matrix[index, index] = by_config[f"w4:{name}"]["calibration_risk"] - base_cal

    task_interactions = {}
    cal_interactions = {}
    interaction_rows = []
    for (first, second) in pairs:
        i = layer_names.index(first)
        j = layer_names.index(second)
        pair_name = f"w4:{first}+{second}"
        task_value = pairwise_interaction(
            base_task,
            by_config[f"w4:{first}"]["task_risk"],
            by_config[f"w4:{second}"]["task_risk"],
            by_config[pair_name]["task_risk"],
        )
        cal_value = pairwise_interaction(
            base_cal,
            by_config[f"w4:{first}"]["calibration_risk"],
            by_config[f"w4:{second}"]["calibration_risk"],
            by_config[pair_name]["calibration_risk"],
        )
        task_matrix[i, j] = task_matrix[j, i] = task_value
        cal_matrix[i, j] = cal_matrix[j, i] = cal_value
        task_interactions[(i, j)] = task_value
        cal_interactions[(i, j)] = cal_value
        interaction_rows.append(
            {
                "layer_i": first,
                "layer_j": second,
                "task_interaction": task_value,
                "calibration_interaction": cal_value,
            }
        )

    valid_task = np.array([row["task_interaction"] for row in interaction_rows])
    valid_cal = np.array([row["calibration_interaction"] for row in interaction_rows])
    if np.std(valid_task) > 0 and np.std(valid_cal) > 0:
        interaction_spearman = float(spearmanr(valid_task, valid_cal).statistic)
    else:
        interaction_spearman = float("nan")
    sign_agreement = float(np.mean(np.sign(valid_task) == np.sign(valid_cal)))

    deltas_task = np.array([task_matrix[i, i] for i in range(len(layer_names))])
    deltas_cal = np.array([cal_matrix[i, i] for i in range(len(layer_names))])
    modules = dict(model.named_modules())
    flash_w8 = np.array(
        [modules[name].weight.numel() * 2.0 for name in layer_names],
        dtype=np.float64,
    )
    flash_budget = float(flash_w8.sum() * 0.85)
    selected_cal = solve_bitwidth_assignment(deltas_cal, cal_interactions, flash_w8, flash_budget)
    selected_task = solve_bitwidth_assignment(deltas_task, task_interactions, flash_w8, flash_budget)
    cal_selection_metrics, cal_bits, _, _ = evaluate_selected_layers(
        model,
        selected_cal,
        layer_names,
        eval_loader,
        device,
        activation_bits=args.activation_bits,
    )
    task_selection_metrics, task_bits, _, _ = evaluate_selected_layers(
        model,
        selected_task,
        layer_names,
        eval_loader,
        device,
        activation_bits=args.activation_bits,
    )

    with (args.output_dir / "interactions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(interaction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(interaction_rows)
    write_config_csv(args.output_dir / "config_metrics.csv", rows)

    plot_interaction_matrices(
        args.output_dir / "interaction_matrices.png",
        layer_names,
        task_matrix,
        cal_matrix,
    )
    plot_config_risks(args.output_dir / "config_risks.png", rows)

    summary = {
        "device": str(device),
        "activation_bits": args.activation_bits,
        "parameter_count": parameter_count,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "history": history,
        "fp32_metrics": fp32_metrics,
        "candidate_layers": layer_names,
        "pairs": pairs,
        "configurations": rows,
        "interaction_spearman": interaction_spearman,
        "interaction_sign_agreement": sign_agreement,
        "calibration_selection": {
            "selected": selected_cal.tolist(),
            "bits": cal_bits,
            "metrics": cal_selection_metrics,
        },
        "task_selection": {
            "selected": selected_task.tolist(),
            "bits": task_bits,
            "metrics": task_selection_metrics,
        },
        "flash_budget": flash_budget,
        "flash_w8": flash_w8.tolist(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    with (args.output_dir / "history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)

    print(json.dumps(
        {
            "device": str(device),
            "activation_bits": args.activation_bits,
            "fp32_acc": fp32_metrics["accuracy"],
            "base_w8_task": base_task,
            "base_w8_cal": base_cal,
            "interaction_spearman": interaction_spearman,
            "sign_agreement": sign_agreement,
            "calibration_selection": cal_bits,
            "task_selection": task_bits,
            "output_dir": str(args.output_dir),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
