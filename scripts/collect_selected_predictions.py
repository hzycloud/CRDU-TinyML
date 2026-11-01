import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.data import build_imagefolder
from cig_mpq.metrics import softmax
from cig_mpq.models import AudioDSCNN, DSCNN
from cig_mpq.quantization import (
    install_activation_quantization_hooks,
    quantize_mixed_precision,
)
from cig_mpq.reproducibility import seed_everything
from cig_mpq.training import evaluate_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["cifar", "speech"], required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def model_and_loader(task: str, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if task == "cifar":
        run_dir = (
            ROOT / "results" / "cifar_pilot_det1"
            if seed == 42
            else ROOT / "results" / f"cifar_pilot_seed{seed}"
        )
        checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
        test_dataset = build_imagefolder(
            checkpoint["args"]["test_root"],
            image_size=32,
            normalize=True,
        )
        loader = DataLoader(
            Subset(test_dataset, list(range(min(3000, len(test_dataset))))),
            batch_size=128,
            shuffle=False,
        )
        model = DSCNN(num_classes=len(checkpoint["classes"]), in_channels=3).to(device)
    else:
        checkpoint = torch.load(
            ROOT / "results" / f"speech_commands_m5_seed{seed}" / "model.pt",
            map_location="cpu",
            weights_only=False,
        )
        data = np.load(
            ROOT / "datasets" / "speech_commands" / "processed" / "kws12_m5" / "test.npz",
            allow_pickle=False,
        )
        features = torch.from_numpy(data["features"]).float().unsqueeze(1)
        labels = torch.from_numpy(data["labels"]).long()
        loader = DataLoader(
            torch.utils.data.TensorDataset(features, labels),
            batch_size=128,
            shuffle=False,
        )
        model = AudioDSCNN(num_classes=12, in_channels=1).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, loader, device


def selected_config(selection: dict) -> str:
    if "config" in selection:
        return selection["config"]
    return selection["selection"]["config"]


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    prediction_dir = result_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    selections = json.loads(
        (result_dir / "all_seed_selections.json").read_text(encoding="utf-8")
    )
    index = {}
    for seed_text, seed_result in selections.items():
        seed = int(seed_text)
        seed_everything(seed)
        model, loader, device = model_and_loader(args.task, seed)
        candidates = {
            row["config"]: row
            for row in json.loads(
                (result_dir / f"seed{seed}_candidates.json").read_text(encoding="utf-8")
            )
        }
        seed_index = {"budgets": {}}
        selected_by_mask = {}
        for budget in seed_result["budgets"]:
            budget_index = {}
            for strategy in ("task_only", "calibration_only", "calibration_constrained"):
                selected = budget.get(strategy)
                if selected is None:
                    budget_index[strategy] = None
                    continue
                config = selected_config(selected)
                candidate = candidates[config]
                budget_index[strategy] = int(candidate["mask"])
                selected_by_mask[int(candidate["mask"])] = candidate
            seed_index["budgets"][str(budget["budget"])] = budget_index

        for mask, candidate in sorted(selected_by_mask.items()):
            quantized = copy.deepcopy(model)
            quantize_mixed_precision(quantized, candidate["bits"], default_bits=8)
            handles = install_activation_quantization_hooks(quantized, bits=8)
            try:
                _, logits, labels = evaluate_logits(quantized, loader, device)
            finally:
                for handle in handles:
                    handle.remove()
            probabilities = softmax(logits / float(candidate["temperature"])).astype(
                np.float32
            )
            np.savez_compressed(
                prediction_dir / f"seed{seed}_mask{mask}.npz",
                probabilities=probabilities,
                labels=labels.astype(np.int64),
                config=np.asarray(candidate["config"]),
                mask=np.asarray(mask, dtype=np.int64),
            )
        index[str(seed)] = seed_index

    (prediction_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"prediction_dir": str(prediction_dir), "seeds": list(index)}, indent=2))


if __name__ == "__main__":
    main()
