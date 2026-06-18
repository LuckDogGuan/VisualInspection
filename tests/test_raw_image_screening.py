import tempfile
import unittest
from pathlib import Path
import sys

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classification.screen_raw_images import (
    RawImageRow,
    build_nice_picture_target,
    collect_raw_images,
    predict_raw_images,
    should_copy_as_nice_picture,
    write_summary,
)


class ConstantModel(torch.nn.Module):
    def forward(self, x):
        logits = torch.zeros((x.shape[0], 11), dtype=torch.float32, device=x.device)
        logits[:, 0] = 1.0
        return logits


class RawImageScreeningTests(unittest.TestCase):
    def test_collect_raw_images_keeps_sources_and_skips_nice_picture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            apspc1 = root / "APSPC1"
            apspc2 = root / "APSPC2"
            nice = root / "nice_picture"
            apspc1.mkdir()
            apspc2.mkdir()
            nice.mkdir()
            Image.new("RGB", (8, 8)).save(apspc1 / "img1.jpg")
            Image.new("RGB", (8, 8)).save(apspc2 / "img2.jpg")
            Image.new("RGB", (8, 8)).save(nice / "skip.jpg")
            (apspc1 / "notes.txt").write_text("skip", encoding="utf-8")

            rows = collect_raw_images(root, source_names=("APSPC1", "APSPC2"))

        self.assertEqual([(row.source_name, row.image_path.name) for row in rows], [("APSPC1", "img1.jpg"), ("APSPC2", "img2.jpg")])

    def test_should_copy_as_nice_picture_only_accepts_top1_normal(self):
        self.assertTrue(should_copy_as_nice_picture({"class_id": 0, "confidence": 0.10}))
        self.assertFalse(should_copy_as_nice_picture({"class_id": 2, "confidence": 0.99}))

    def test_build_nice_picture_target_preserves_source_and_adds_prediction(self):
        source = Path("data/raw_images/APSPC1/img123.jpg")
        target = build_nice_picture_target(
            output_root=Path("data/raw_images/nice_picture"),
            source_name="APSPC1",
            image_path=source,
            class_id=0,
            class_name="normal",
            confidence=0.94321,
        )

        self.assertEqual(target.parent.as_posix(), "data/raw_images/nice_picture/APSPC1")
        self.assertEqual(target.name, "img123__pred_00_normal__conf_0.943.jpg")

    def test_write_summary_includes_ascii_keys_for_server_logs(self):
        records = [
            {"source_name": "APSPC1", "class_id": 0, "confidence": 0.9},
            {"source_name": "APSPC2", "class_id": 2, "confidence": 0.8},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_md = Path(tmp) / "summary.md"

            write_summary(records, output_md, copied_count=1, normal_class_id=0, min_confidence=0.0)

            content = output_md.read_text(encoding="utf-8")

        self.assertIn("total_images=2", content)
        self.assertIn("copied_to_nice_picture=1", content)
        self.assertIn("normal_class_id=0", content)

    def test_predict_raw_images_records_corrupt_images_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.jpg"
            bad = root / "bad.jpg"
            Image.new("RGB", (8, 8)).save(good)
            bad.write_bytes(b"not a real jpg")

            records = predict_raw_images(
                ConstantModel(),
                [RawImageRow(good, "APSPC1"), RawImageRow(bad, "APSPC1")],
                image_size=8,
                device=torch.device("cpu"),
            )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(records[1]["status"], "error")
        self.assertIn("bad.jpg", records[1]["source_path"])
        self.assertTrue(records[1]["error"])


if __name__ == "__main__":
    unittest.main()
