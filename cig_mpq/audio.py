import wave
from pathlib import Path

import numpy as np


def read_pcm16_mono_wav(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1:
            raise ValueError(f"{path} is not mono")
        if stream.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        raw = stream.readframes(stream.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def pad_or_truncate(waveform: np.ndarray, length: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size >= length:
        return waveform[:length].copy()
    output = np.zeros(length, dtype=np.float32)
    output[: waveform.size] = waveform
    return output


def window_waveform(waveform: np.ndarray, length: int, seed: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size <= length:
        return pad_or_truncate(waveform, length)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, waveform.size - length + 1))
    return waveform[start : start + length].copy()


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(
    sample_rate: int,
    fft_size: int,
    num_filters: int = 40,
    lower_hz: float = 20.0,
    upper_hz: float = 4000.0,
) -> np.ndarray:
    mel_points = np.linspace(
        _hz_to_mel(lower_hz),
        _hz_to_mel(upper_hz),
        num_filters + 2,
    )
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((num_filters, fft_size // 2 + 1), dtype=np.float64)
    for index in range(num_filters):
        left, center, right = bins[index : index + 3]
        if center > left:
            filters[index, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)
        if right > center:
            filters[index, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)
    return filters


def mfcc_features(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    clip_samples: int = 16000,
    window_samples: int = 480,
    stride_samples: int = 320,
    fft_size: int = 512,
    mel_filters: int = 40,
    coefficients: int = 10,
) -> np.ndarray:
    waveform = pad_or_truncate(np.asarray(waveform, dtype=np.float32), clip_samples)
    emphasized = np.empty_like(waveform)
    emphasized[0] = waveform[0]
    emphasized[1:] = waveform[1:] - 0.97 * waveform[:-1]
    frame_count = 1 + (clip_samples - window_samples) // stride_samples
    frames = np.stack(
        [
            emphasized[index * stride_samples : index * stride_samples + window_samples]
            for index in range(frame_count)
        ]
    )
    windowed = frames * np.hanning(window_samples)
    spectrum = np.fft.rfft(windowed, n=fft_size, axis=1)
    power = (np.abs(spectrum) ** 2) / fft_size
    filterbank = _mel_filterbank(sample_rate, fft_size, num_filters=mel_filters)
    log_mel = np.log(np.maximum(power @ filterbank.T, 1e-10))

    n = np.arange(mel_filters)[None, :]
    k = np.arange(coefficients)[:, None]
    basis = np.sqrt(2.0 / mel_filters) * np.cos(
        np.pi * (n + 0.5) * k / mel_filters
    )
    basis[0, :] = 1.0 / np.sqrt(mel_filters)
    return (log_mel @ basis.T).astype(np.float32)
