import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cig_mpq.audio import (
    mfcc_features,
    pad_or_truncate,
    read_pcm16_mono_wav,
    window_waveform,
)


class AudioTest(unittest.TestCase):
    def test_read_pcm16_mono_wav_returns_float_signal(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.wav"
            samples = np.array([-32768, 0, 16384, 32767], dtype=np.int16)
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(samples.tobytes())

            waveform = read_pcm16_mono_wav(path)

        self.assertEqual(waveform.dtype, np.float32)
        np.testing.assert_allclose(
            waveform,
            np.array([-1.0, 0.0, 0.5, 32767 / 32768], dtype=np.float32),
        )

    def test_pad_or_truncate_returns_requested_length(self):
        short = np.array([1.0, 2.0], dtype=np.float32)
        long = np.arange(10, dtype=np.float32)

        np.testing.assert_array_equal(
            pad_or_truncate(short, 4),
            np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            pad_or_truncate(long, 4),
            np.arange(4, dtype=np.float32),
        )

    def test_mfcc_features_have_mlperf_compatible_shape(self):
        sample_rate = 16000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        waveform = np.sin(2 * np.pi * 440.0 * time).astype(np.float32)

        features = mfcc_features(waveform, sample_rate=sample_rate)

        self.assertEqual(features.shape, (49, 10))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_window_waveform_uses_seeded_random_crop_for_long_background(self):
        waveform = np.arange(24000, dtype=np.float32)

        first = window_waveform(waveform, length=16000, seed=5)
        second = window_waveform(waveform, length=16000, seed=5)
        other = window_waveform(waveform, length=16000, seed=6)

        self.assertEqual(first.shape, (16000,))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        self.assertFalse(np.array_equal(first, waveform[:16000]))


if __name__ == "__main__":
    unittest.main()
