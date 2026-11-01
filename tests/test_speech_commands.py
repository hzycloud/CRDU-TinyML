import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cig_mpq.speech_commands import (
    TARGET_WORDS,
    build_speech_commands_manifest,
)


class SpeechCommandsManifestTest(unittest.TestCase):
    def _create_fixture(self, root: Path) -> None:
        for word in TARGET_WORDS:
            directory = root / word
            directory.mkdir()
            (directory / f"{word}_train.wav").write_bytes(b"")
            (directory / f"{word}_a.wav").write_bytes(b"")
            (directory / f"{word}_b.wav").write_bytes(b"")
        unknown = root / "bed"
        unknown.mkdir()
        (unknown / "bed_train.wav").write_bytes(b"")
        (unknown / "bed_a.wav").write_bytes(b"")
        (unknown / "bed_b.wav").write_bytes(b"")
        background = root / "_background_noise_"
        background.mkdir()
        (background / "noise.wav").write_bytes(b"")

        validation = [
            f"{word}/{word}_a.wav" for word in TARGET_WORDS
        ] + ["bed/bed_a.wav"]
        testing = [
            f"{word}/{word}_b.wav" for word in TARGET_WORDS
        ] + ["bed/bed_b.wav"]
        (root / "validation_list.txt").write_text(
            "\n".join(validation) + "\n",
            encoding="utf-8",
        )
        (root / "testing_list.txt").write_text(
            "\n".join(testing) + "\n",
            encoding="utf-8",
        )

    def test_manifest_uses_official_split_and_adds_unknown_and_silence(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._create_fixture(root)

            rows = build_speech_commands_manifest(
                root,
                split="validation",
                seed=7,
                samples_per_class=None,
            )

        labels = sorted({row["label"] for row in rows})
        self.assertEqual(labels, list(range(12)))
        self.assertTrue(
            all(Path(row["path"]).name.endswith("_a.wav") for row in rows if row["kind"] == "word")
        )
        self.assertEqual(sum(row["kind"] == "silence" for row in rows), 1)
        self.assertEqual(sum(row["label"] == 11 for row in rows), 1)

    def test_manifest_caps_each_class_deterministically(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._create_fixture(root)

            first = build_speech_commands_manifest(
                root,
                split="train",
                seed=9,
                samples_per_class=1,
            )
            second = build_speech_commands_manifest(
                root,
                split="train",
                seed=9,
                samples_per_class=1,
            )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(sorted({row["label"] for row in first}), list(range(12)))


if __name__ == "__main__":
    unittest.main()
