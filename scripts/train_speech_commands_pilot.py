import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.metrics import classification_metrics
from cig_mpq.models import AudioDSCNN
from cig_mpq.reproducibility import seed_everything
from cig_mpq.speech_commands import TARGET_WORDS
from cig_mpq.temperature import fit_temperature
from cig_mpq.training import evaluate_logits


CLASS_NAMES = list(TARGET_WORDS) + ["silence", "unknown"]


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
        default=ROOT / "results" / "speech_commands_pilot",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split(processed_dir: Path, split: str) -> TensorDataset:
    archive = np.load(processed_dir / f"{split}.npz", allow_pickle=False)
    features = torch.from_numpy(archive["features"]).float().unsqueeze(1)
    labels = torch.from_numpy(archive["labels"]).long()
    return TensorDataset(features, labels)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    for features, labels in tqdm(loader, desc="train", leave=False):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.shape[0]
        seen += labels.shape[0]
    return running_loss / seen


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = load_split(args.processed_dir, "train")
    validation_dataset = load_split(args.processed_dir, "validation")
    test_dataset = load_split(args.processed_dir, "test")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = AudioDSCNN(num_classes=12, in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_accuracy = -1.0
    best_state = None
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        validation_metrics, _, _ = evaluate_logits(model, validation_loader, device)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_metrics": validation_metrics,
            }
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(json.dumps(history[-1], ensure_ascii=False))

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_metrics, validation_logits, validation_labels = evaluate_logits(
        model,
        validation_loader,
        device,
    )
    temperature = fit_temperature(validation_logits, validation_labels)
    raw_test_metrics, test_logits, test_labels = evaluate_logits(
        model,
        test_loader,
        device,
    )
    calibrated_test_metrics = classification_metrics(
        test_logits / temperature,
        test_labels,
    )
    elapsed = time.time() - started

    torch.save(
        {
            "state_dict": best_state,
            "classes": CLASS_NAMES,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        args.output_dir / "model.pt",
    )
    result = {
        "device": str(device),
        "elapsed_seconds": elapsed,
        "best_validation_accuracy": best_accuracy,
        "temperature": float(temperature),
        "validation_metrics": validation_metrics,
        "test_raw_metrics": raw_test_metrics,
        "test_calibrated_metrics": calibrated_test_metrics,
        "history": history,
        "record_counts": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
