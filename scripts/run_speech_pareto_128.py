import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_speech_quantization_interactions import evaluate_configuration, load_npz
from cig_mpq.experiment import choose_candidate_layers
from cig_mpq.models import AudioDSCNN
from cig_mpq.pareto import pareto_frontier, select_for_budget
from cig_mpq.reproducibility import seed_everything


STRATEGIES = ("task_only", "calibration_only", "calibration_constrained")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "datasets" / "speech_commands" / "processed" / "kws12_pilot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "speech_pareto_128",
    )
    parser.add_argument("--search-size", type=int, default=600)
    parser.add_argument("--temperature-size", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--model-path-template",
        default="results/speech_commands_m5_seed{seed}/model.pt",
    )
    return parser.parse_args()


def bits_from_mask(mask: int, layer_names: list[str]) -> dict[str, int]:
    return {
        name: 4 if mask & (1 << index) else 8
        for index, name in enumerate(layer_names)
    }


def config_name(mask: int, bits: dict[str, int], layer_names: list[str]) -> str:
    pattern = "".join("4" if bits[name] == 4 else "8" for name in layer_names)
    return f"mask:{pattern}:{mask}"


def candidate_weight_bytes(
    model: torch.nn.Module,
    bits: dict[str, int],
    layer_names: list[str],
) -> int:
    modules = dict(model.named_modules())
    return int(
        sum(
            (modules[name].weight.numel() * bits[name] + 7) // 8
            for name in layer_names
        )
    )


def selection_view(row: dict) -> dict:
    search = row["search_calibrated"]
    return {
        "config": row["config"],
        "mask": row["mask"],
        "task_risk": row["search_raw_nll"],
        "accuracy": search["accuracy"],
        "ece": search["ece"],
        "nll": search["nll"],
        "brier": search["brier"],
        "calibration_risk": search["nll"] + 0.5 * search["brier"],
        "weight_bytes": row["weight_bytes"],
        "flash_fraction": row["flash_fraction"],
    }


def selection_output(view: dict | None, rows_by_config: dict[str, dict]) -> dict | None:
    if view is None:
        return None
    row = rows_by_config[view["config"]]
    return {
        "selection": view,
        "test_raw_metrics": row["test_raw_metrics"],
        "test_calibrated_metrics": row["test_calibrated_metrics"],
        "temperature": row["temperature"],
    }


def write_candidates_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "mask",
        "config",
        "bits",
        "weight_bytes",
        "flash_fraction",
        "search_accuracy",
        "search_task_risk",
        "search_ece",
        "search_calibration_risk",
        "test_accuracy",
        "test_ece",
        "test_nll",
        "test_brier",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            search = row["search_calibrated"]
            test = row["test_calibrated_metrics"]
            writer.writerow(
                {
                    "mask": row["mask"],
                    "config": row["config"],
                    "bits": "".join(str(row["bits"][name]) for name in row["layer_names"]),
                    "weight_bytes": row["weight_bytes"],
                    "flash_fraction": row["flash_fraction"],
                    "search_accuracy": search["accuracy"],
                    "search_task_risk": row["search_raw_nll"],
                    "search_ece": search["ece"],
                    "search_calibration_risk": search["nll"] + 0.5 * search["brier"],
                    "test_accuracy": test["accuracy"],
                    "test_ece": test["ece"],
                    "test_nll": test["nll"],
                    "test_brier": test["brier"],
                }
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation = load_npz(args.processed_dir / "validation.npz")
    test_dataset = load_npz(args.processed_dir / "test.npz")
    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(len(validation), generator=generator).tolist()
    search_dataset = Subset(validation, permutation[: args.search_size])
    temperature_dataset = Subset(
        validation,
        permutation[
            args.search_size : args.search_size + args.temperature_size
        ],
    )
    search_loader = DataLoader(search_dataset, batch_size=args.batch_size, shuffle=False)
    temperature_loader = DataLoader(
        temperature_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    all_selections = {}
    for seed in args.seeds:
        model_path = ROOT / args.model_path_template.format(seed=seed)
        if not model_path.exists():
            raise FileNotFoundError(f"missing model for seed {seed}: {model_path}")
        seed_everything(seed)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model = AudioDSCNN(num_classes=12, in_channels=1).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        layer_names = choose_candidate_layers(model, max_layers=999)
        total_masks = 1 << len(layer_names)
        masks = list(range(total_masks))
        if args.max_configs > 0:
            masks = masks[: args.max_configs]

        uniform_w8_bytes = candidate_weight_bytes(
            model,
            {name: 8 for name in layer_names},
            layer_names,
        )
        rows = []
        started = time.time()
        for mask in tqdm(masks, desc=f"seed {seed} masks"):
            bits = bits_from_mask(mask, layer_names)
            metrics = evaluate_configuration(
                model,
                bits,
                search_loader,
                temperature_loader,
                test_loader,
                device,
                args.activation_bits,
            )
            measured_bytes = candidate_weight_bytes(model, bits, layer_names)
            row = {
                "seed": seed,
                "mask": mask,
                "config": config_name(mask, bits, layer_names),
                "bits": bits,
                "layer_names": layer_names,
                "weight_bytes": measured_bytes,
                "flash_fraction": measured_bytes / uniform_w8_bytes,
                "activation_bits": args.activation_bits,
                "temperature": metrics["temperature"],
                "search_raw_nll": metrics["task_risk"],
                "search_calibrated": metrics["interaction_calibrated_metrics"],
                "test_raw_metrics": metrics["test_raw_metrics"],
                "test_calibrated_metrics": metrics["test_calibrated_metrics"],
            }
            row["selection_view"] = selection_view(row)
            rows.append(row)

        rows_by_config = {row["config"]: row for row in rows}
        baseline = rows_by_config[
            config_name(0, {name: 8 for name in layer_names}, layer_names)
        ]["selection_view"]
        selection_views = [row["selection_view"] for row in rows]
        budget_results = []
        for budget in (1.0, 0.95, 0.90, 0.85, 0.80):
            selected = select_for_budget(selection_views, budget, baseline)
            budget_results.append(
                {
                    "budget": budget,
                    "accuracy_floor": selected["accuracy_floor"],
                    "ece_cap": selected["ece_cap"],
                    **{
                        strategy: selection_output(
                            selected[strategy],
                            rows_by_config,
                        )
                        for strategy in STRATEGIES
                    },
                }
            )

        seed_result = {
            "seed": seed,
            "model_path": str(model_path.relative_to(ROOT)),
            "layer_names": layer_names,
            "candidate_count": len(rows),
            "uniform_w8_candidate_bytes": uniform_w8_bytes,
            "elapsed_seconds": time.time() - started,
            "budgets": budget_results,
            "search_frontier": pareto_frontier(
                selection_views,
                objectives=("weight_bytes", "task_risk", "calibration_risk"),
            ),
        }
        write_candidates_csv(args.output_dir / f"seed{seed}_candidates.csv", rows)
        (args.output_dir / f"seed{seed}_candidates.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (args.output_dir / f"seed{seed}_selection.json").write_text(
            json.dumps(seed_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        all_selections[seed] = seed_result

    (args.output_dir / "all_seed_selections.json").write_text(
        json.dumps(all_selections, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "candidate_counts": {
                    seed: result["candidate_count"]
                    for seed, result in all_selections.items()
                },
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
