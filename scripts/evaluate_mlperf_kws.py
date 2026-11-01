import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.metrics import classification_metrics
from cig_mpq.mlperf_kws import (
    dequantize_tflite_output,
    load_kws_binary_directory,
    prepare_tflite_input,
)


BENCHMARK = ROOT / "third_party" / "mlperf_tiny_zip" / "tiny-master" / "benchmark"
MODELS = {
    "float32": (
        BENCHMARK
        / "training"
        / "keyword_spotting"
        / "trained_models"
        / "kws_ref_model_float32.tflite"
    ),
    "int8": (
        BENCHMARK
        / "training"
        / "keyword_spotting"
        / "trained_models"
        / "kws_ref_model.tflite"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=ROOT / "datasets" / "kws01_official",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "mlperf_kws",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def data_manifest_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest().upper()


def evaluate_model(
    model_name: str,
    model_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict, list[dict]]:
    feature_interpreter = Interpreter(model_path=str(MODELS["int8"]))
    feature_interpreter.allocate_tensors()
    feature_quantization = feature_interpreter.get_input_details()[0]["quantization"]
    source_scale = float(feature_quantization[0])
    source_zero_point = int(feature_quantization[1])
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    predictor_rows = []
    probabilities_rows = []

    for feature, label, sample_index in tqdm(
        zip(features, labels, range(len(labels))),
        total=len(labels),
        desc=f"MLPerf KWS {model_name}",
    ):
        model_input = prepare_tflite_input(
            feature.reshape(1, 49, 10, 1),
            dtype=input_detail["dtype"],
            scale=float(input_detail["quantization"][0]),
            zero_point=int(input_detail["quantization"][1]),
            source_scale=source_scale,
            source_zero_point=source_zero_point,
        )
        interpreter.set_tensor(input_detail["index"], model_input)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_detail["index"])
        probabilities = dequantize_tflite_output(
            raw_output,
            dtype=output_detail["dtype"],
            scale=float(output_detail["quantization"][0]),
            zero_point=int(output_detail["quantization"][1]),
        ).reshape(-1)
        probabilities = probabilities / probabilities.sum()
        probabilities_rows.append(probabilities)
        predicted = int(np.argmax(probabilities))
        predictor_rows.append(
            {
                "sample_index": sample_index,
                "true_label": int(label),
                "predicted_label": predicted,
                "confidence": float(probabilities[predicted]),
                "correct": int(predicted == int(label)),
            }
        )

    probability_matrix = np.stack(probabilities_rows)
    log_probabilities = np.log(np.clip(probability_matrix, 1e-12, 1.0))
    metrics = classification_metrics(log_probabilities, labels)
    result = {
        "model": model_name,
        "model_path": str(model_path.relative_to(ROOT)),
        "model_sha256": sha256(model_path),
        "sample_count": int(len(labels)),
        "input_shape": [int(value) for value in input_detail["shape"]],
        "input_dtype": str(input_detail["dtype"].__name__),
        "output_dtype": str(output_detail["dtype"].__name__),
        "metrics": metrics,
        "note": (
            "Official kws01 1,000-sample test set only; this is not a "
            "training, interaction, or MCU deployment result."
        ),
    }
    return result, predictor_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, labels, names = load_kws_binary_directory(args.features_dir)
    paths = [args.features_dir / name for name in names]
    outputs = {}
    for model_name, model_path in MODELS.items():
        result, predictions = evaluate_model(
            model_name,
            model_path,
            features,
            labels,
        )
        outputs[model_name] = result
        with (args.output_dir / f"{model_name}_predictions.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "sample_index",
                    "true_label",
                    "predicted_label",
                    "confidence",
                    "correct",
                ],
            )
            writer.writeheader()
            writer.writerows(predictions)

    summary = {
        "dataset": {
            "name": "MLPerf Tiny kws01",
            "source": "https://github.com/eembc/energyrunner/tree/main/datasets/kws01",
            "feature_count": int(len(features)),
            "feature_shape": list(features.shape[1:]),
            "bytes": int(sum(path.stat().st_size for path in paths)),
            "sha256_manifest": data_manifest_hash(paths),
            "label_counts": {
                str(label): int(np.sum(labels == label))
                for label in sorted(set(labels.tolist()))
            },
        },
        "models": outputs,
    }
    output_path = args.output_dir / "full_eval.json"
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
