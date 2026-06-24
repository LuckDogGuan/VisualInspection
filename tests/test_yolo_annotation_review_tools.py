from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.apply_yolo_annotation_review_decisions import build_clean_dataset
from scripts.create_yolo_annotation_review_guides import YoloBox
from scripts.create_yolo_prediction_review import PredBox, analyze_pair, box_iou
from scripts.create_yolo_duplicate_review import DuplicateCandidate, choose_duplicate_keep, group_duplicate_candidates
from scripts.auto_optimize_yolo_labels_conservative import (
    optimize_boxes_for_class,
)
from scripts.create_stain_manual_review_pack import (
    build_pack,
    select_stain_rows,
    worker_template_fields,
)


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


def test_duplicate_review_prefers_batch_005_then_newer_action():
    candidates = [
        DuplicateCandidate(
            row_index=0,
            image_name="stain_batch_001_0001.jpg",
            class_name="stain",
            batch_id="batch_001",
            source_key="same-source",
            hash_value="0" * 64,
            action="keep",
            image_path="a.jpg",
            label_path="a.txt",
        ),
        DuplicateCandidate(
            row_index=1,
            image_name="stain_batch_005_0001.jpg",
            class_name="stain",
            batch_id="batch_005",
            source_key="same-source",
            hash_value="0" * 64,
            action="merge_candidate",
            image_path="b.jpg",
            label_path="b.txt",
        ),
    ]

    assert choose_duplicate_keep(candidates).batch_id == "batch_005"


def test_duplicate_review_groups_by_source_or_near_hash():
    candidates = [
        DuplicateCandidate(0, "a.jpg", "stain", "batch_001", "same", "0" * 64, "keep", "a.jpg", "a.txt"),
        DuplicateCandidate(1, "b.jpg", "stain", "batch_005", "same", "1" * 64, "keep", "b.jpg", "b.txt"),
        DuplicateCandidate(2, "c.jpg", "dent", "batch_002", "unique-c", "0" * 64, "keep", "c.jpg", "c.txt"),
        DuplicateCandidate(3, "d.jpg", "dent", "batch_005", "unique-d", "0" * 63 + "1", "keep", "d.jpg", "d.txt"),
    ]

    groups = group_duplicate_candidates(candidates, hash_threshold=2)
    grouped_names = [sorted(item.image_name for item in group) for group in groups]

    assert ["a.jpg", "b.jpg"] in grouped_names
    assert ["c.jpg", "d.jpg"] in grouped_names


def test_conservative_optimizer_merges_dense_stain_boxes():
    boxes = [
        YoloBox(2, 0.20, 0.30, 0.04, 0.04),
        YoloBox(2, 0.25, 0.31, 0.04, 0.04),
        YoloBox(2, 0.30, 0.32, 0.04, 0.04),
        YoloBox(2, 0.35, 0.33, 0.04, 0.04),
    ]

    optimized, action, _ = optimize_boxes_for_class("stain", boxes)

    assert action == "stain_merge_dense_regions"
    assert len(optimized) < len(boxes)
    assert optimized[0].class_id == 2


def test_conservative_optimizer_keeps_broad_stain_clusters_unmerged():
    boxes = [
        YoloBox(2, 0.10, 0.25, 0.04, 0.04),
        YoloBox(2, 0.30, 0.25, 0.04, 0.04),
        YoloBox(2, 0.50, 0.25, 0.04, 0.04),
        YoloBox(2, 0.70, 0.25, 0.04, 0.04),
    ]

    optimized, action, _ = optimize_boxes_for_class("stain", boxes)

    assert action == "kept_original"
    assert optimized == boxes


def test_conservative_optimizer_merges_overlapping_dent_only():
    boxes = [
        YoloBox(0, 0.50, 0.50, 0.08, 0.08),
        YoloBox(0, 0.52, 0.50, 0.08, 0.08),
        YoloBox(0, 0.80, 0.80, 0.05, 0.05),
    ]

    optimized, action, _ = optimize_boxes_for_class("dent", boxes)

    assert action == "dent_merge_duplicate_boxes"
    assert len(optimized) == 2


def test_conservative_optimizer_shrinks_wide_crack_box():
    boxes = [YoloBox(3, 0.50, 0.50, 0.90, 0.20)]

    optimized, action, _ = optimize_boxes_for_class("crack", boxes)

    assert action == "crack_shrink_tall_wide_boxes"
    assert optimized[0].width == boxes[0].width
    assert optimized[0].height < boxes[0].height


def test_select_stain_rows_keeps_priority_order_and_limit():
    rows = [
        {"image_name": "a.jpg", "class_name": "dent", "score": "30"},
        {"image_name": "b.jpg", "class_name": "stain", "score": "20"},
        {"image_name": "c.jpg", "class_name": "stain", "score": "10"},
    ]

    selected = select_stain_rows(rows, limit=1)

    assert [row["image_name"] for row in selected] == ["b.jpg"]


def test_worker_template_fields_include_original_and_skip_reason():
    fields = worker_template_fields()

    assert "worker_judgement" in fields
    assert "skip_reason" in fields
    assert "original_image" in fields
    assert "reference_guide" in fields


def test_stain_manual_pack_copies_original_images_and_worker_template(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    prediction_review_dir = tmp_path / "review"
    output_dir = tmp_path / "out"
    image_name = "stain_batch_001_0001.jpg"
    guide_name = "guides/stain_batch_001_0001_review.jpg"

    write_image(dataset_root / "images" / "val" / image_name)
    write_image(prediction_review_dir / guide_name)
    with (prediction_review_dir / "review_priority.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_name",
                "class_name",
                "score",
                "issue",
                "gt_count",
                "pred_count",
                "matched_count",
                "max_iou",
                "guide_image",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_name": image_name,
                "class_name": "stain",
                "score": "10",
                "issue": "漏检 1",
                "gt_count": "1",
                "pred_count": "0",
                "matched_count": "0",
                "max_iou": "0",
                "guide_image": guide_name,
            }
        )

    rows = build_pack(
        prediction_review_dir=prediction_review_dir,
        output_dir=output_dir,
        dataset_root=dataset_root,
        limit=5,
    )

    assert rows[0]["local_original"].startswith("originals/")
    assert (output_dir / rows[0]["local_original"]).exists()
    worker_template = (output_dir / "qc_worker_reply_template.csv").read_text(encoding="utf-8-sig")
    assert "worker_judgement" in worker_template
    assert image_name in worker_template
    assert "originals/" in worker_template
    assert (output_dir / "qc_worker_index.html").exists()

    send_dir = output_dir / "发给质检工人"
    assert (send_dir / "01_质检说明.md").exists()
    assert (send_dir / "02_质检回复模板.csv").exists()
    chinese_template = (send_dir / "02_质检回复模板.csv").read_text(encoding="utf-8-sig")
    assert "判断结果" in chinese_template
    assert "处理方式" in chinese_template
    assert "跳过原因" in chinese_template
    assert (send_dir / "原图" / f"01_{image_name}").exists()
