import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.data import build_imagefolder
from cig_mpq.metrics import classification_metrics
from cig_mpq.models import DSCNN
from cig_mpq.pareto import (
    add_calibration_risk,
    pareto_frontier,
    select_for_budget,
)
from cig_mpq.quantization import install_activation_quantization_hooks, quantize_mixed_precision
from cig_mpq.reproducibility import seed_everything
from cig_mpq.temperature import fit_temperature
from cig_mpq.training import evaluate_logits


def model_path_for_seed(seed: int) -> Path:
    if seed == 42:
        return ROOT / "results" / "cifar_pilot_det1" / "model.pt"
    return ROOT / "results" / f"cifar_pilot_seed{seed}" / "model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "pareto_pilot",
    )
    parser.add_argument("--search-limit", type=int, default=1000)
    parser.add_argument("--temperature-limit", type=int, default=1000)
    parser.add_argument("--eval-limit", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--max-configs", type=int, default=0)
    return parser.parse_args()


def data_loaders(checkpoint: dict, args: argparse.Namespace):
    model_args = checkpoint["args"]
    full_train = build_imagefolder(model_args["train_root"], image_size=32, normalize=True)
    test_dataset = build_imagefolder(model_args["test_root"], image_size=32, normalize=True)

    generator = torch.Generator().manual_seed(int(model_args["seed"]))
    shuffled = torch.randperm(len(full_train), generator=generator).tolist()
    train_count = int(model_args["train_limit"])
    available_calibration = int(model_args["temp_limit"])
    required = args.search_limit + args.temperature_limit
    if required > available_calibration:
        raise ValueError(
            f"requested {required} search/calibration samples but only "
            f"{available_calibration} are available"
        )
    calibration_indices = shuffled[train_count:train_count + available_calibration]
    search_subset = Subset(full_train, calibration_indices[:args.search_limit])
    temperature_subset = Subset(
        full_train,
        calibration_indices[
            args.search_limit:args.search_limit + args.temperature_limit
        ],
    )
    eval_subset = Subset(
        test_dataset,
        list(range(min(args.eval_limit, len(test_dataset)))),
    )
    return (
        DataLoader(search_subset, batch_size=args.batch_size, shuffle=False),
        DataLoader(temperature_subset, batch_size=args.batch_size, shuffle=False),
        DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False),
    )


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DSCNN(num_classes=len(checkpoint["classes"]), in_channels=3)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint


def bits_from_mask(mask: int, layer_names: list[str]) -> dict[str, int]:
    return {
        name: 4 if mask & (1 << index) else 8
        for index, name in enumerate(layer_names)
    }


def config_name(bits: dict[str, int], layer_names: list[str]) -> str:
    pattern = "".join("4" if bits[name] == 4 else "8" for name in layer_names)
    selected = [name for name in layer_names if bits[name] == 4]
    if not selected:
        return f"mask:{pattern}:uniform_w8"
    return f"mask:{pattern}:w4:" + "+".join(selected)


def weight_bytes(model: torch.nn.Module, bits: dict[str, int], default_bits: int = 8) -> int:
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            layer_bits = bits.get(name, default_bits)
            total += (module.weight.numel() * layer_bits + 7) // 8
    return int(total)


def evaluate_candidate(
    model: torch.nn.Module,
    bits: dict[str, int],
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    device: torch.device,
    activation_bits: int,
) -> dict:
    search_loader, temperature_loader, eval_loader = loaders
    candidate = copy.deepcopy(model)
    quantize_mixed_precision(candidate, bits, default_bits=8)
    handles = install_activation_quantization_hooks(candidate, bits=activation_bits)
    try:
        search_raw, search_logits, search_labels = evaluate_logits(
            candidate,
            search_loader,
            device,
        )
        _, temperature_logits, temperature_labels = evaluate_logits(
            candidate,
            temperature_loader,
            device,
        )
        eval_raw, eval_logits, eval_labels = evaluate_logits(
            candidate,
            eval_loader,
            device,
        )
    finally:
        for handle in handles:
            handle.remove()

    temperature = fit_temperature(temperature_logits, temperature_labels)
    search_calibrated = add_calibration_risk(
        classification_metrics(
            search_logits / temperature,
            search_labels,
        )
    )
    eval_calibrated = add_calibration_risk(
        classification_metrics(
            eval_logits / temperature,
            eval_labels,
        )
    )
    return {
        "temperature": float(temperature),
        "search_raw": search_raw,
        "search_calibrated": search_calibrated,
        "eval_raw": eval_raw,
        "eval_calibrated": eval_calibrated,
    }


def selection_view(row: dict) -> dict:
    search = row["search_calibrated"]
    return {
        "config": row["config"],
        "task_risk": row["search_raw"]["nll"],
        "accuracy": search["accuracy"],
        "ece": search["ece"],
        "nll": search["nll"],
        "brier": search["brier"],
        "calibration_risk": search["calibration_risk"],
        "weight_bytes": row["weight_bytes"],
        "flash_fraction": row["flash_fraction"],
    }


def write_seed_outputs(
    output_dir: Path,
    seed: int,
    rows: list[dict],
    selections: dict,
) -> None:
    fields = [
        "config",
        "mask",
        "bits",
        "weight_bytes",
        "flash_fraction",
        "temperature",
        "search_accuracy",
        "search_ece",
        "search_task_risk",
        "search_calibration_risk",
        "eval_accuracy",
        "eval_ece",
        "eval_nll",
        "eval_brier",
    ]
    with (output_dir / f"seed{seed}_candidates.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            search = row["search_calibrated"]
            evaluation = row["eval_calibrated"]
            writer.writerow(
                {
                    "config": row["config"],
                    "mask": row["mask"],
                    "bits": "".join(str(bits) for bits in row["bits"].values()),
                    "weight_bytes": row["weight_bytes"],
                    "flash_fraction": row["flash_fraction"],
                    "temperature": row["temperature"],
                    "search_accuracy": search["accuracy"],
                    "search_ece": search["ece"],
                    "search_task_risk": row["search_raw"]["nll"],
                    "search_calibration_risk": search["calibration_risk"],
                    "eval_accuracy": evaluation["accuracy"],
                    "eval_ece": evaluation["ece"],
                    "eval_nll": evaluation["nll"],
                    "eval_brier": evaluation["brier"],
                }
            )
    (output_dir / f"seed{seed}_candidates.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / f"seed{seed}_selection.json").write_text(
        json.dumps(selections, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_selections = {}

    for seed in args.seeds:
        model_path = model_path_for_seed(seed)
        if not model_path.exists():
            raise FileNotFoundError(f"no trained model found for seed {seed}: {model_path}")
        seed_everything(seed)
        model, checkpoint = load_model(model_path, device)
        layer_names = json.loads(
            (model_path.parent / "summary.json").read_text(encoding="utf-8")
        )["candidate_layers"]
        loaders = data_loaders(checkpoint, args)
        uniform_w8 = {name: 8 for name in layer_names}
        uniform_bytes = weight_bytes(model, uniform_w8)
        total_masks = 1 << len(layer_names)
        masks = list(range(total_masks))
        if args.max_configs > 0:
            masks = masks[:args.max_configs]

        rows = []
        started = time.time()
        for mask in tqdm(masks, desc=f"seed {seed} configurations"):
            bits = bits_from_mask(mask, layer_names)
            row = evaluate_candidate(
                model,
                bits,
                loaders,
                device,
                activation_bits=args.activation_bits,
            )
            measured_bytes = weight_bytes(model, bits)
            row.update(
                {
                    "seed": seed,
                    "mask": mask,
                    "config": config_name(bits, layer_names),
                    "bits": bits,
                    "weight_bytes": measured_bytes,
                    "flash_fraction": measured_bytes / uniform_bytes,
                    "layer_names": layer_names,
                    "activation_bits": args.activation_bits,
                }
            )
            row["selection"] = selection_view(row)
            rows.append(row)

        if not rows:
            raise RuntimeError("no candidate configurations were evaluated")
        baseline = min(
            rows,
            key=lambda row: abs(row["flash_fraction"] - 1.0),
        )["selection"]
        selections = {
            "seed": seed,
            "uniform_w8_weight_bytes": uniform_bytes,
            "elapsed_seconds": time.time() - started,
            "candidate_count": len(rows),
            "baseline": baseline,
            "budgets": [
                select_for_budget(
                    [row["selection"] for row in rows],
                    budget,
                    baseline,
                )
                for budget in (1.0, 0.95, 0.90, 0.85)
            ],
            "search_objective_frontier": pareto_frontier(
                [row["selection"] for row in rows],
                objectives=("weight_bytes", "task_risk", "calibration_risk"),
            ),
            "eval_descriptive_frontier": pareto_frontier(
                [
                    {
                        "config": row["config"],
                        "weight_bytes": row["weight_bytes"],
                        "eval_nll": row["eval_calibrated"]["nll"],
                        "eval_calibration_risk": (
                            row["eval_calibrated"]["nll"]
                            + 0.5 * row["eval_calibrated"]["brier"]
                        ),
                    }
                    for row in rows
                ],
                objectives=(
                    "weight_bytes",
                    "eval_nll",
                    "eval_calibration_risk",
                ),
            ),
        }
        write_seed_outputs(args.output_dir, seed, rows, selections)
        all_selections[str(seed)] = selections

    (args.output_dir / "all_seed_selections.json").write_text(
        json.dumps(all_selections, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "seeds": args.seeds,
                "candidate_counts": {
                    seed: values["candidate_count"]
                    for seed, values in all_selections.items()
                },
                "output_dir": str(args.output_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
