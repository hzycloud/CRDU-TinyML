import argparse
import copy
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.experiment import (
    build_quantization_configs,
    choose_candidate_layers,
    risk_delta_and_interactions,
)
from cig_mpq.metrics import classification_metrics
from cig_mpq.models import AudioDSCNN
from cig_mpq.quantization import (
    install_activation_quantization_hooks,
    quantize_mixed_precision,
)
from cig_mpq.reproducibility import seed_everything
from cig_mpq.search import solve_bitwidth_assignment
from cig_mpq.temperature import fit_temperature
from cig_mpq.training import evaluate_logits


MODEL_RUNS = {
    42: ROOT / "results" / "speech_commands_pilot" / "model.pt",
    43: ROOT / "results" / "speech_commands_pilot_seed43" / "model.pt",
    44: ROOT / "results" / "speech_commands_pilot_seed44" / "model.pt",
}


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
        default=ROOT / "results" / "speech_quantization_interactions",
    )
    parser.add_argument("--interaction-size", type=int, default=600)
    parser.add_argument("--temperature-size", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--flash-fraction", type=float, default=0.85)
    return parser.parse_args()


def load_npz(path: Path) -> TensorDataset:
    data = np.load(path, allow_pickle=False)
    features = torch.from_numpy(data["features"]).float().unsqueeze(1)
    labels = torch.from_numpy(data["labels"]).long()
    return TensorDataset(features, labels)


def evaluate_configuration(
    model: torch.nn.Module,
    bits: dict[str, int],
    interaction_loader: DataLoader,
    temperature_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    activation_bits: int,
) -> dict:
    candidate = copy.deepcopy(model)
    quantize_mixed_precision(candidate, bits, default_bits=8)
    handles = install_activation_quantization_hooks(candidate, bits=activation_bits)
    try:
        interaction_raw, interaction_logits, interaction_labels = evaluate_logits(
            candidate,
            interaction_loader,
            device,
        )
        _, temperature_logits, temperature_labels = evaluate_logits(
            candidate,
            temperature_loader,
            device,
        )
        test_raw, test_logits, test_labels = evaluate_logits(
            candidate,
            test_loader,
            device,
        )
    finally:
        for handle in handles:
            handle.remove()
    temperature = fit_temperature(temperature_logits, temperature_labels)
    interaction_calibrated = classification_metrics(
        interaction_logits / temperature,
        interaction_labels,
    )
    test_calibrated = classification_metrics(
        test_logits / temperature,
        test_labels,
    )
    return {
        "temperature": float(temperature),
        "task_risk": float(interaction_raw["nll"]),
        "calibration_risk": float(
            interaction_calibrated["nll"] + 0.5 * interaction_calibrated["brier"]
        ),
        "ece": float(interaction_calibrated["ece"]),
        "brier": float(interaction_calibrated["brier"]),
        "interaction_raw_metrics": interaction_raw,
        "interaction_calibrated_metrics": interaction_calibrated,
        "test_raw_metrics": test_raw,
        "test_calibrated_metrics": test_calibrated,
    }


def evaluate_selection(
    model: torch.nn.Module,
    selected: np.ndarray,
    layer_names: list[str],
    temperature_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    activation_bits: int,
) -> tuple[dict[str, int], dict]:
    bits = {
        name: 4
        for name, flag in zip(layer_names, selected.tolist())
        if flag == 1
    }
    candidate = copy.deepcopy(model)
    quantize_mixed_precision(candidate, bits, default_bits=8)
    handles = install_activation_quantization_hooks(candidate, bits=activation_bits)
    try:
        _, temperature_logits, temperature_labels = evaluate_logits(
            candidate,
            temperature_loader,
            device,
        )
        test_raw, test_logits, test_labels = evaluate_logits(
            candidate,
            test_loader,
            device,
        )
    finally:
        for handle in handles:
            handle.remove()
    temperature = fit_temperature(temperature_logits, temperature_labels)
    return bits, {
        "temperature": float(temperature),
        "test_raw_metrics": test_raw,
        "test_calibrated_metrics": classification_metrics(
            test_logits / temperature,
            test_labels,
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation = load_npz(args.processed_dir / "validation.npz")
    test_dataset = load_npz(args.processed_dir / "test.npz")
    if args.interaction_size + args.temperature_size > len(validation):
        raise ValueError("interaction and temperature sizes exceed validation set")
    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(len(validation), generator=generator).tolist()
    interaction_dataset = Subset(validation, permutation[: args.interaction_size])
    temperature_dataset = Subset(
        validation,
        permutation[
            args.interaction_size : args.interaction_size + args.temperature_size
        ],
    )
    loaders = {
        "interaction": DataLoader(
            interaction_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        ),
        "temperature": DataLoader(
            temperature_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        ),
    }

    all_results = {}
    for seed, model_path in MODEL_RUNS.items():
        seed_everything(seed)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model = AudioDSCNN(num_classes=12, in_channels=1).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        layer_names = choose_candidate_layers(model, max_layers=len(list(model.modules())))
        pairs = list(itertools.combinations(layer_names, 2))
        configs = build_quantization_configs(layer_names, pairs)

        metrics_by_config = {}
        rows = []
        for config_name, bits in tqdm(
            configs.items(),
            desc=f"seed {seed} configurations",
        ):
            metrics = evaluate_configuration(
                model,
                bits,
                loaders["interaction"],
                loaders["temperature"],
                loaders["test"],
                device,
                args.activation_bits,
            )
            metrics_by_config[config_name] = metrics
            rows.append(
                {
                    "config": config_name,
                    "task_risk": metrics["task_risk"],
                    "calibration_risk": metrics["calibration_risk"],
                    "ece": metrics["ece"],
                    "brier": metrics["brier"],
                    "test_accuracy": metrics["test_calibrated_metrics"]["accuracy"],
                    "test_ece": metrics["test_calibrated_metrics"]["ece"],
                    "test_nll": metrics["test_calibrated_metrics"]["nll"],
                    "test_brier": metrics["test_calibrated_metrics"]["brier"],
                }
            )

        task_deltas, task_interactions = risk_delta_and_interactions(
            metrics_by_config,
            layer_names,
            pairs,
            "task_risk",
        )
        cal_deltas, cal_interactions = risk_delta_and_interactions(
            metrics_by_config,
            layer_names,
            pairs,
            "calibration_risk",
        )
        task_vector = np.asarray([task_interactions[key] for key in sorted(task_interactions)])
        cal_vector = np.asarray([cal_interactions[key] for key in sorted(cal_interactions)])
        if np.std(task_vector) > 0 and np.std(cal_vector) > 0:
            interaction_spearman = float(spearmanr(task_vector, cal_vector).statistic)
        else:
            interaction_spearman = float("nan")
        sign_agreement = float(np.mean(np.sign(task_vector) == np.sign(cal_vector)))

        modules = dict(model.named_modules())
        flash_w8 = np.asarray(
            [modules[name].weight.numel() * 2.0 for name in layer_names],
            dtype=np.float64,
        )
        flash_budget = float(flash_w8.sum() * args.flash_fraction)
        selections = {}
        for label, deltas, interactions in (
            ("task_loss", task_deltas, task_interactions),
            ("calibration_risk", cal_deltas, cal_interactions),
        ):
            selected = solve_bitwidth_assignment(
                np.asarray(deltas),
                interactions,
                flash_w8,
                flash_budget,
            )
            bits, evaluation = evaluate_selection(
                model,
                selected,
                layer_names,
                loaders["temperature"],
                loaders["test"],
                device,
                args.activation_bits,
            )
            selections[label] = {
                "selected": selected.tolist(),
                "bits": bits,
                "evaluation": evaluation,
            }

        with (args.output_dir / f"seed{seed}_configs.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        all_results[seed] = {
            "model_path": str(model_path.relative_to(ROOT)),
            "candidate_layers": layer_names,
            "pair_count": len(pairs),
            "configuration_count": len(configs),
            "activation_bits": args.activation_bits,
            "flash_budget": flash_budget,
            "flash_w8": flash_w8.tolist(),
            "task_interactions": [
                {"pair": [layer_names[i], layer_names[j]], "value": task_interactions[(i, j)]}
                for i, j in sorted(task_interactions)
            ],
            "calibration_interactions": [
                {"pair": [layer_names[i], layer_names[j]], "value": cal_interactions[(i, j)]}
                for i, j in sorted(cal_interactions)
            ],
            "interaction_spearman": interaction_spearman,
            "interaction_sign_agreement": sign_agreement,
            "selections": selections,
        }

    spearman_values = [value["interaction_spearman"] for value in all_results.values()]
    sign_values = [value["interaction_sign_agreement"] for value in all_results.values()]
    summary = {
        "seeds": list(all_results),
        "median_spearman": float(np.median(spearman_values)),
        "median_sign_agreement": float(np.median(sign_values)),
        "results": all_results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "median_spearman": summary["median_spearman"],
                "median_sign_agreement": summary["median_sign_agreement"],
                "selections": {
                    seed: {
                        name: value["bits"]
                        for name, value in result["selections"].items()
                    }
                    for seed, result in all_results.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
