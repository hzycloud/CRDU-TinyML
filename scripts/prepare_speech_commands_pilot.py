import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.audio import (
    mfcc_features,
    pad_or_truncate,
    read_pcm16_mono_wav,
    window_waveform,
)
from cig_mpq.speech_commands import TARGET_WORDS, build_speech_commands_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "datasets" / "speech_commands" / "extracted",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets" / "speech_commands" / "processed" / "kws12_pilot",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--validation-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def waveform_for_row(row: dict) -> np.ndarray:
    waveform = read_pcm16_mono_wav(row["path"])
    if row["kind"] == "silence":
        return window_waveform(
            waveform,
            length=16000,
            seed=int(row["sample_seed"]),
        )
    return pad_or_truncate(waveform, 16000)


def row_features(row: dict) -> np.ndarray:
    waveform = waveform_for_row(row)
    return mfcc_features(waveform)


def materialize_rows(rows: list[dict], workers: int) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        features = list(executor.map(row_features, rows))
    return np.stack(features).astype(np.float32)


def build_split(
    root: Path,
    split: str,
    per_class: int,
    seed: int,
    workers: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows = build_speech_commands_manifest(
        root,
        split=split,
        seed=seed,
        samples_per_class=per_class,
    )
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    if sorted(set(labels.tolist())) != list(range(12)):
        raise ValueError(f"{split} split does not contain all 12 classes")
    features = materialize_rows(rows, workers)
    return rows, features, labels


def normalize(
    train: np.ndarray,
    others: list[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = np.maximum(train.std(axis=0, keepdims=True), 1e-5)
    normalized = [(train - mean) / std]
    normalized.extend((array - mean) / std for array in others)
    return normalized, mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_limits = {
        "train": args.train_per_class,
        "validation": args.validation_per_class,
        "test": args.test_per_class,
    }

    split_rows = {}
    split_features = {}
    split_labels = {}
    for split, limit in split_limits.items():
        rows, features, labels = build_split(
            args.dataset_root,
            split,
            limit,
            args.seed,
            args.workers,
        )
        split_rows[split] = rows
        split_features[split] = features
        split_labels[split] = labels

    normalized, mean, std = normalize(
        split_features["train"],
        [split_features["validation"], split_features["test"]],
    )
    for split, features in zip(split_limits, normalized):
        np.savez_compressed(
            args.output_dir / f"{split}.npz",
            features=features.astype(np.float32),
            labels=split_labels[split],
            paths=np.asarray([row["path"] for row in split_rows[split]]),
        )

    np.savez(
        args.output_dir / "normalization.npz",
        mean=mean,
        std=std,
    )
    with (args.output_dir / "manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "split",
                "path",
                "label",
                "class_name",
                "kind",
                "sample_seed",
            ],
        )
        writer.writeheader()
        for split, rows in split_rows.items():
            for row in rows:
                writer.writerow({"split": split, **row})

    summary = {
        "dataset_root": str(args.dataset_root),
        "seed": args.seed,
        "target_words": list(TARGET_WORDS),
        "silence_label": 10,
        "unknown_label": 11,
        "splits": {
            split: {
                "records": len(split_labels[split]),
                "feature_shape": list(split_features[split].shape),
                "class_counts": {
                    str(label): int(count)
                    for label, count in sorted(Counter(split_labels[split].tolist()).items())
                },
            }
            for split in split_limits
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
