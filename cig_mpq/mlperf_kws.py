import re
from pathlib import Path

import numpy as np


def prepare_tflite_input(
    features: np.ndarray,
    dtype: type,
    scale: float,
    zero_point: int,
    source_scale: float | None = None,
    source_zero_point: int | None = None,
) -> np.ndarray:
    values = np.asarray(features)
    if (source_scale is None) != (source_zero_point is None):
        raise ValueError("source scale and zero point must be provided together")
    if source_scale is not None:
        values = (
            values.astype(np.float64) - int(source_zero_point)
        ) * float(source_scale)
    if not np.issubdtype(dtype, np.integer):
        return values.astype(dtype)
    if scale <= 0.0:
        raise ValueError("quantized TFLite input requires a positive scale")
    limits = np.iinfo(dtype)
    quantized = np.rint(values / float(scale) + int(zero_point))
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def dequantize_tflite_output(
    output: np.ndarray,
    dtype: type,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    values = np.asarray(output)
    if not np.issubdtype(dtype, np.integer):
        return values.astype(np.float64)
    return (values.astype(np.float64) - int(zero_point)) * float(scale)


def label_from_kws_filename(filename: str) -> int:
    label = int(Path(filename).stem.rsplit("_", 1)[-1])
    if label < 0 or label >= 12:
        raise ValueError(f"invalid KWS label in filename: {filename}")
    return label


def load_kws_binary_directory(
    directory: str | Path,
    rows: int = 49,
    columns: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    directory = Path(directory)
    paths = sorted(directory.glob("*.bin"))
    if not paths:
        raise ValueError(f"no KWS binary files found in {directory}")

    expected_bytes = rows * columns
    features = []
    labels = []
    names = []
    for path in paths:
        raw = path.read_bytes()
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{path.name} has {len(raw)} bytes; expected {expected_bytes}"
            )
        features.append(np.frombuffer(raw, dtype=np.int8).copy())
        labels.append(label_from_kws_filename(path.name))
        names.append(path.name)
    return (
        np.stack(features).reshape(len(paths), rows, columns),
        np.asarray(labels, dtype=np.int64),
        names,
    )


def parse_int8_2d_array(source: str, rows: int, columns: int) -> np.ndarray:
    name_index = source.find("g_kws_inputs")
    if name_index == -1:
        raise ValueError("g_kws_inputs declaration not found")
    start = source.find("=", name_index)
    end = source.find("};", start)
    if start == -1 or end == -1 or end <= start:
        raise ValueError("C array initializer not found")
    body = source[start + 1 : end]
    values = [int(value) for value in re.findall(r"-?\d+", body)]
    expected = rows * columns
    if len(values) != expected:
        raise ValueError(f"expected {expected} values, found {len(values)}")
    return np.asarray(values, dtype=np.int8).reshape(rows, columns)


def parse_kws_labels(source: str) -> tuple[np.ndarray, list[str]]:
    labels = []
    names = []
    for line in source.splitlines():
        line = line.strip()
        if not line:
            continue
        filename, category_count, label = line.split(",")
        if int(category_count) != 12:
            raise ValueError(f"unexpected category count for {filename}")
        names.append(filename)
        labels.append(int(label))
    return np.asarray(labels, dtype=np.int64), names

