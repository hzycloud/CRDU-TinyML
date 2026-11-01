import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cig_mpq.data import build_imagefolder


class DataTest(unittest.TestCase):
    def test_imagefolder_preserves_class_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for class_name, color in (("cat", "red"), ("dog", "blue")):
                class_dir = root / class_name
                class_dir.mkdir()
                Image.new("RGB", (32, 32), color=color).save(class_dir / "sample.png")

            dataset = build_imagefolder(root, image_size=32)

            self.assertEqual(len(dataset), 2)
            self.assertEqual(sorted(dataset.classes), ["cat", "dog"])
            image, label = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 32, 32))
            self.assertIn(label, (0, 1))


if __name__ == "__main__":
    unittest.main()
