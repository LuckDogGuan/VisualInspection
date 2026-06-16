import tempfile
import unittest
import warnings
from pathlib import Path
import sys

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classification.config import CLASS_NAME_TO_ID, CLASS_ID_TO_CN, get_config
from classification.data import LabelRow, build_label_rows, build_test_rows, limit_rows_by_class, split_rows
from classification.exporting import export_torchscript_checkpoint
from classification.inference import resolve_inference_image_size
from classification.modeling import build_classifier
from classification.train import FixedRotation
from classification.visualize_defects import draw_box, imread_unicode, imwrite_unicode
from classification.evaluate_model import (
    sample_labeled_rows_by_class,
    sample_unlabeled_rows,
    safe_folder_name,
)


class ClassificationPipelineTests(unittest.TestCase):
    def test_label_mapping_matches_requirement(self):
        self.assertEqual(CLASS_NAME_TO_ID["Clean sample"], 0)
        self.assertEqual(CLASS_NAME_TO_ID["Dirty spot"], 10)
        self.assertEqual(CLASS_ID_TO_CN[2], "擦花")
        self.assertEqual(len(CLASS_NAME_TO_ID), 11)

    def test_build_label_rows_ignores_isolated_and_test_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (8, 8)).save(root / "clean.jpg")
            class_dir = root / "scuffing"
            class_dir.mkdir()
            Image.new("RGB", (8, 8)).save(class_dir / "a.jpg")
            rare_dir = root / "新增少样本缺陷"
            rare_dir.mkdir()
            Image.new("RGB", (8, 8)).save(rare_dir / "rare.jpg")
            test_dir = root / "测试文件_未标注"
            test_dir.mkdir()
            Image.new("RGB", (8, 8)).save(test_dir / "0.jpg")

            rows = build_label_rows(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].label, 2)
        self.assertEqual(rows[0].class_name, "scuffing")

    def test_build_test_rows_reads_only_test_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = root / "测试文件_未标注"
            test_dir.mkdir()
            Image.new("RGB", (8, 8)).save(test_dir / "0.jpg")
            (test_dir / "notes.txt").write_text("skip", encoding="utf-8")

            rows = build_test_rows(root)

        self.assertEqual([row.image_path.name for row in rows], ["0.jpg"])

    def test_split_rows_is_deterministic_and_keeps_all_rows(self):
        rows = [
            LabelRow(image_path=Path(f"{idx}.jpg"), label=idx % 2, class_name=str(idx % 2))
            for idx in range(20)
        ]

        train_a, val_a = split_rows(rows, val_ratio=0.2, seed=123)
        train_b, val_b = split_rows(rows, val_ratio=0.2, seed=123)

        self.assertEqual([row.image_path for row in train_a], [row.image_path for row in train_b])
        self.assertEqual([row.image_path for row in val_a], [row.image_path for row in val_b])
        self.assertEqual(len(train_a) + len(val_a), 20)
        self.assertEqual(len(val_a), 4)

    def test_limit_rows_by_class_keeps_classes_balanced(self):
        rows = [
            LabelRow(image_path=Path(f"{label}-{idx}.jpg"), label=label, class_name=str(label))
            for label in range(3)
            for idx in range(5)
        ]

        limited = limit_rows_by_class(rows, max_samples=6)

        self.assertEqual(len(limited), 6)
        self.assertEqual({row.label for row in limited}, {0, 1, 2})
        self.assertEqual([row.image_path.name for row in limited], ["0-0.jpg", "1-0.jpg", "2-0.jpg", "0-1.jpg", "1-1.jpg", "2-1.jpg"])

    def test_limit_rows_by_class_does_not_duplicate_when_classes_are_uneven(self):
        rows = [
            LabelRow(image_path=Path(f"0-{idx}.jpg"), label=0, class_name="0")
            for idx in range(5)
        ]
        rows.append(LabelRow(image_path=Path("1-0.jpg"), label=1, class_name="1"))

        limited = limit_rows_by_class(rows, max_samples=5)

        names = [row.image_path.name for row in limited]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names, ["0-0.jpg", "1-0.jpg", "0-1.jpg", "0-2.jpg", "0-3.jpg"])

    def test_config_sets_linux_gpu_visibility(self):
        config = get_config("linux")
        self.assertEqual(config.gpu_ids, "0")
        self.assertEqual(config.classification_root.as_posix(), "data/ali2018")
        self.assertEqual(config.batch_size, 96)
        self.assertEqual(config.workers, 16)
        self.assertTrue(config.use_amp)
        self.assertEqual(config.prefetch_factor, 4)

    def test_config_sets_windows_local_data_root(self):
        config = get_config("windows")
        self.assertEqual(config.classification_root.as_posix(), "data/ali2018")
        self.assertEqual(config.batch_size, 96)
        self.assertEqual(config.workers, 0)
        self.assertTrue(config.use_amp)

    def test_config_auto_selects_platform_defaults(self):
        config = get_config("auto")
        self.assertIn(config.env, {"windows", "linux"})
        self.assertEqual(config.classification_root.as_posix(), "data/ali2018")

    def test_model_output_shape(self):
        model = build_classifier(num_classes=11, architecture="resnet18", pretrained=False)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 11))

    def test_export_torchscript_checkpoint_writes_callable_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "checkpoint.pth.tar"
            output_dir = root / "artifact"
            model = build_classifier(num_classes=11, architecture="resnet18", pretrained=False)
            torch.save(
                {
                    "architecture": "resnet18",
                    "num_classes": 11,
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            artifact_path, labels_path = export_torchscript_checkpoint(checkpoint_path, output_dir, image_size=64)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.jit")
                scripted = torch.jit.load(str(artifact_path), map_location="cpu")
            scripted.eval()
            with torch.no_grad():
                output = scripted(torch.zeros(1, 3, 64, 64))
            self.assertTrue(labels_path.exists())

        self.assertEqual(tuple(output.shape), (1, 11))
        self.assertEqual(artifact_path.name, "classifier.torchscript.pt")

    def test_resolve_inference_image_size_reads_deploy_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "classifier.torchscript.pt"
            model_path.write_bytes(b"placeholder")
            (Path(tmp) / "labels.json").write_text('{"image_size": 96}', encoding="utf-8")

            image_size = resolve_inference_image_size(model_path, default_image_size=384)

        self.assertEqual(image_size, 96)

    def test_fixed_rotation_keeps_image_size_for_right_angle_augmentation(self):
        image = Image.new("RGB", (8, 4), color="white")
        transform = FixedRotation((0,))

        rotated = transform(image)

        self.assertEqual(rotated.size, (8, 4))

    def test_visualization_helpers_support_chinese_paths_and_labels(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "中文图片.jpg"
            output_path = Path(tmp) / "输出图片.jpg"
            image = np.zeros((80, 120, 3), dtype=np.uint8)
            self.assertTrue(imwrite_unicode(image_path, image))

            loaded = imread_unicode(image_path)
            draw_box(loaded, [10, 20, 80, 60], "擦花")
            self.assertTrue(imwrite_unicode(output_path, loaded))
            reread = imread_unicode(output_path)

        self.assertEqual(tuple(reread.shape), (80, 120, 3))
        self.assertGreater(int(reread.sum()), 0)

    def test_sample_labeled_rows_by_class_keeps_up_to_limit_per_class(self):
        rows = [
            LabelRow(image_path=Path(f"{label}-{idx}.jpg"), label=label, class_name=str(label))
            for label in range(2)
            for idx in range(5)
        ]

        sampled = sample_labeled_rows_by_class(rows, per_class=3, seed=123)

        counts = {}
        for row in sampled:
            counts[row.label] = counts.get(row.label, 0) + 1
        self.assertEqual(counts, {0: 3, 1: 3})

    def test_sample_unlabeled_rows_limits_total_count(self):
        from classification.data import TestRow

        rows = [TestRow(image_path=Path(f"{idx}.jpg")) for idx in range(10)]

        sampled = sample_unlabeled_rows(rows, count=4, seed=123)

        self.assertEqual(len(sampled), 4)
        self.assertEqual(len({row.image_path for row in sampled}), 4)

    def test_safe_folder_name_removes_path_separators(self):
        self.assertEqual(safe_folder_name(0, "正常/合格品"), "00_正常_合格品")


if __name__ == "__main__":
    unittest.main()
