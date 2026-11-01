import json
import sys
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.mlperf_kws import parse_int8_2d_array


BENCHMARK = ROOT / "third_party" / "mlperf_tiny_zip" / "tiny-master" / "benchmark"


def main() -> None:
    inputs_path = (
        BENCHMARK
        / "reference_submissions"
        / "keyword_spotting"
        / "kws"
        / "kws_input_data.cc"
    )
    model_path = (
        BENCHMARK
        / "training"
        / "keyword_spotting"
        / "trained_models"
        / "kws_ref_model_float32.tflite"
    )
    features = parse_int8_2d_array(inputs_path.read_text(encoding="utf-8"), rows=1, columns=490)
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    model_input = features[0].reshape(1, 49, 10, 1).astype(input_detail["dtype"])
    interpreter.set_tensor(input_detail["index"], model_input)
    interpreter.invoke()
    probabilities = interpreter.get_tensor(output_detail["index"])[0]
    result = {
        "input_shape": [int(value) for value in input_detail["shape"]],
        "input_dtype": str(input_detail["dtype"].__name__),
        "output_shape": [int(value) for value in output_detail["shape"]],
        "output_dtype": str(output_detail["dtype"].__name__),
        "predicted_class": int(np.argmax(probabilities)),
        "predicted_probability": float(np.max(probabilities)),
        "note": "single-sample smoke test; no accuracy claim",
    }
    output_dir = ROOT / "results" / "mlperf_kws"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
