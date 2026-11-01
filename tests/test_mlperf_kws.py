import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cig_mpq.mlperf_kws import (
    dequantize_tflite_output,
    load_kws_binary_directory,
    parse_int8_2d_array,
    parse_kws_labels,
    prepare_tflite_input,
)


class MLPerfKwsTest(unittest.TestCase):
    def test_parse_int8_2d_array(self):
        source = """
        // =============================================================================
        const int8_t g_kws_inputs[2][3] = {
          {1, -2, 3},
          {-4, 5, 6}
        };
        """

        array = parse_int8_2d_array(source, rows=2, columns=3)

        np.testing.assert_array_equal(
            array,
            np.array([[1, -2, 3], [-4, 5, 6]], dtype=np.int8),
        )

    def test_parse_kws_labels(self):
        source = "sample_0.bin,12,7\nsample_1.bin,12,2\n"

        labels, names = parse_kws_labels(source)

        np.testing.assert_array_equal(labels, np.array([7, 2], dtype=np.int64))
        self.assertEqual(names, ["sample_0.bin", "sample_1.bin"])

    def test_load_kws_binary_directory_reads_features_and_filename_labels(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first = np.arange(490, dtype=np.int16).astype(np.int8)
            second = -first
            (directory / "tst_000001_Left_2.bin").write_bytes(first.tobytes())
            (directory / "tst_000000_Stop_7.bin").write_bytes(second.tobytes())

            features, labels, names = load_kws_binary_directory(directory)

        self.assertEqual(features.shape, (2, 49, 10))
        np.testing.assert_array_equal(features[0].reshape(-1), second)
        np.testing.assert_array_equal(features[1].reshape(-1), first)
        np.testing.assert_array_equal(labels, np.array([7, 2], dtype=np.int64))
        self.assertEqual(
            names,
            ["tst_000000_Stop_7.bin", "tst_000001_Left_2.bin"],
        )

    def test_load_kws_binary_directory_rejects_wrong_file_size(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "tst_000000_Stop_7.bin").write_bytes(b"\x00" * 489)

            with self.assertRaises(ValueError):
                load_kws_binary_directory(directory)

    def test_prepare_tflite_input_quantizes_and_clips_int8(self):
        features = np.array([0.0, 20.0, -20.0], dtype=np.float32)

        prepared = prepare_tflite_input(
            features,
            dtype=np.int8,
            scale=0.1,
            zero_point=0,
        )

        np.testing.assert_array_equal(
            prepared,
            np.array([0, 127, -128], dtype=np.int8),
        )

    def test_prepare_tflite_input_dequantizes_published_int8_features_for_float_model(self):
        published_features = np.array([-83, -82, 0], dtype=np.int8)

        prepared = prepare_tflite_input(
            published_features,
            dtype=np.float32,
            scale=0.0,
            zero_point=0,
            source_scale=0.5,
            source_zero_point=-83,
        )

        np.testing.assert_allclose(prepared, np.array([0.0, 0.5, 41.5]))

    def test_prepare_tflite_input_preserves_published_int8_features_for_int8_model(self):
        published_features = np.array([-83, -82, 0], dtype=np.int8)

        prepared = prepare_tflite_input(
            published_features,
            dtype=np.int8,
            scale=0.5,
            zero_point=-83,
            source_scale=0.5,
            source_zero_point=-83,
        )

        np.testing.assert_array_equal(
            prepared,
            np.array([-83, -82, 0], dtype=np.int8),
        )

    def test_dequantize_tflite_output_recovers_float_values(self):
        output = np.array([-128, 0, 127], dtype=np.int8)

        dequantized = dequantize_tflite_output(
            output,
            dtype=np.int8,
            scale=0.5,
            zero_point=-128,
        )

        np.testing.assert_allclose(dequantized, np.array([0.0, 64.0, 127.5]))


if __name__ == "__main__":
    unittest.main()
