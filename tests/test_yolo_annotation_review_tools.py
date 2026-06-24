from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.apply_yolo_annotation_review_decisions import build_clean_dataset
from scripts.create_yolo_annotation_review_guides import YoloBox
from scripts.create_yolo_prediction_review import PredBox, analyze_pair, box_iou


def write_image(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), (180, 180, 180)).save(path)


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_clean_dataset_merges_problem_batch_and_preserves_chinese_manifest(tmp_path: Path):
    stage_root = tmp_path / "data" / "yolo_stage3_manual"
    batch_root = stage_root / "batches" / "batch_004"
    image_path = batch_root / "images" / "stain__脏点样本.jpg"
    label_path = batch_root / "labels" / "stain__脏点样本.txt"
    write_image(image_path)
    write_label(
        label_path,
        [
            "2 0.20 0.30 0.05 0.05",
            "2 0.24 0.31 0.05 0.05",
            "2 0.28 0.32 0.05 0.05",
        ],
    )
    registry_path = stage_root / "annotation_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_id",
                "status",
                "class",
                "true_class_cn",
                "confidence",
                "source_path",
                "batch_image_path",
                "batch_label_path",
                "selected_at",
                "annotated_at",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "batch_id": "batch_004",
                "status": "annotated",
                "class": "stain",
                "true_class_cn": "脏点",
                "confidence": "0.9",
                "source_path": str(tmp_path / "源图" / "脏点样本.jpg"),
                "batch_image_path": str(image_path),
                "batch_label_path": str(label_path),
                "selected_at": "",
                "annotated_at": "",
                "notes": "",
            }
        )

    output_root = build_clean_dataset(
        stage_root=stage_root,
        output_name="unit_clean",
        problem_batches={"batch_004"},
        problem_classes={"stain"},
        val_ratio=0.0,
        seed=1,
    )

    exported_labels = list((output_root / "labels" / "train").glob("*.txt"))
    assert len(exported_labels) == 1
    exported_text = exported_labels[0].read_text(encoding="utf-8")
    assert len([line for line in exported_text.splitlines() if line.strip()]) == 1
    assert exported_labels[0].name == "stain_batch_004_0001.txt"

    manifest = (output_root / "clean_manifest.csv").read_text(encoding="utf-8-sig")
    assert "脏点样本.jpg" in manifest
    assert "merge_candidate" in manifest


def test_build_clean_dataset_keeps_non_problem_class_labels_unchanged(tmp_path: Path):
    stage_root = tmp_path / "data" / "yolo_stage3_manual"
    batch_root = stage_root / "batches" / "batch_001"
    image_path = batch_root / "images" / "crack__开裂样本.jpg"
    label_path = batch_root / "labels" / "crack__开裂样本.txt"
    write_image(image_path)
    original = "3 0.50 0.50 0.20 0.10\n3 0.30 0.40 0.10 0.05\n"
    write_label(label_path, original.splitlines())
    registry_path = stage_root / "annotation_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_id",
                "status",
                "class",
                "true_class_cn",
                "confidence",
                "source_path",
                "batch_image_path",
                "batch_label_path",
                "selected_at",
                "annotated_at",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "batch_id": "batch_001",
                "status": "annotated",
                "class": "crack",
                "true_class_cn": "涂层开裂",
                "confidence": "0.9",
                "source_path": str(tmp_path / "源图" / "开裂样本.jpg"),
                "batch_image_path": str(image_path),
                "batch_label_path": str(label_path),
                "selected_at": "",
                "annotated_at": "",
                "notes": "",
            }
        )

    output_root = build_clean_dataset(
        stage_root=stage_root,
        output_name="unit_clean",
        problem_batches={"batch_004"},
        problem_classes={"stain"},
        val_ratio=0.0,
        seed=1,
    )

    exported_labels = list((output_root / "labels" / "train").glob("*.txt"))
    assert len(exported_labels) == 1
    assert exported_labels[0].read_text(encoding="utf-8") == original


def test_build_clean_dataset_applies_manual_review_decision_keep_old(tmp_path: Path):
    stage_root = tmp_path / "data" / "yolo_stage3_manual"
    batch_root = stage_root / "batches" / "batch_004"
    image_path = batch_root / "images" / "stain__人工决策样本.jpg"
    label_path = batch_root / "labels" / "stain__人工决策样本.txt"
    write_image(image_path)
    write_label(
        label_path,
        [
            "2 0.20 0.30 0.05 0.05",
            "2 0.24 0.31 0.05 0.05",
            "2 0.28 0.32 0.05 0.05",
        ],
    )
    registry_path = stage_root / "annotation_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_id",
                "status",
                "class",
                "true_class_cn",
                "confidence",
                "source_path",
                "batch_image_path",
                "batch_label_path",
                "selected_at",
                "annotated_at",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "batch_id": "batch_004",
                "status": "annotated",
                "class": "stain",
                "true_class_cn": "脏点",
                "confidence": "0.9",
                "source_path": str(tmp_path / "源图" / "人工决策样本.jpg"),
                "batch_image_path": str(image_path),
                "batch_label_path": str(label_path),
                "selected_at": "",
                "annotated_at": "",
                "notes": "",
            }
        )
    decisions_path = stage_root / "review" / "manual_review_decisions.csv"
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with decisions_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["batch_label_path", "decision", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "batch_label_path": str(label_path),
                "decision": "keep_old:3",
                "notes": "只保留第三个原始框",
            }
        )

    output_root = build_clean_dataset(
        stage_root=stage_root,
        output_name="unit_clean",
        problem_batches={"batch_004"},
        problem_classes={"stain"},
        val_ratio=0.0,
        seed=1,
        decisions_path=decisions_path,
    )

    exported_labels = list((output_root / "labels" / "train").glob("*.txt"))
    assert len(exported_labels) == 1
    assert exported_labels[0].read_text(encoding="utf-8") == "2 0.280000 0.320000 0.050000 0.050000\n"
    manifest = (output_root / "clean_manifest.csv").read_text(encoding="utf-8-sig")
    assert "manual:keep_old:3" in manifest


def test_prediction_review_box_iou():
    left = YoloBox(0, 0.5, 0.5, 0.4, 0.4)
    right = YoloBox(0, 0.5, 0.5, 0.4, 0.4)
    shifted = YoloBox(0, 0.7, 0.5, 0.4, 0.4)

    assert math.isclose(box_iou(left, right), 1.0)
    assert 0.32 < box_iou(left, shifted) < 0.34


def test_prediction_review_scores_missing_and_duplicate_boxes():
    gt = [PredBox(YoloBox(0, 0.5, 0.5, 0.2, 0.2))]
    missing_score, missing_issue, missing_matched, _ = analyze_pair("dent.jpg", gt, [], 0.35)
    duplicate_score, duplicate_issue, duplicate_matched, _ = analyze_pair(
        "dent.jpg",
        gt,
        [
            PredBox(YoloBox(0, 0.5, 0.5, 0.2, 0.2), 0.9),
            PredBox(YoloBox(0, 0.52, 0.5, 0.2, 0.2), 0.6),
        ],
        0.35,
    )

    assert "漏检" in missing_issue
    assert missing_score > duplicate_score
    assert missing_matched == 0
    assert "重复框" in duplicate_issue
    assert duplicate_matched == 1
