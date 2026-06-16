from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.config import CLASS_ID_TO_CN, get_config, resolve_path
from classification.data import LabelRow, TestRow, build_label_rows, build_test_rows
from classification.inference import load_model_for_inference, predict_image, resolve_inference_image_size
from classification.visualize_defects import draw_box, imread_unicode, imwrite_unicode, load_text_font


def safe_folder_name(class_id: int, class_name_cn: str) -> str:
    safe_name = class_name_cn.replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"{class_id:02d}_{safe_name}"


def sample_labeled_rows_by_class(rows: Sequence[LabelRow], per_class: int, seed: int) -> list[LabelRow]:
    grouped: dict[int, list[LabelRow]] = defaultdict(list)
    for row in rows:
        grouped[row.label].append(row)

    rng = random.Random(seed)
    sampled: list[LabelRow] = []
    for label in sorted(grouped):
        class_rows = list(grouped[label])
        rng.shuffle(class_rows)
        sampled.extend(class_rows[:per_class])
    return sampled


def sample_unlabeled_rows(rows: Sequence[TestRow], count: int, seed: int) -> list[TestRow]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def load_yolo_detector(weights_path: Path | None):
    if weights_path is None or not weights_path.exists():
        return None
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def predict_boxes(detector, image) -> list[list[float]]:
    if detector is None:
        return []
    results = detector.predict(source=image, conf=0.25, verbose=False)
    boxes: list[list[float]] = []
    for result in results:
        for box in result.boxes:
            boxes.append(box.xyxy[0].tolist())
    return boxes


def draw_header(image, lines: Sequence[str]) -> None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    font = load_text_font(size=24)
    line_height = 32
    height = line_height * len(lines) + 10
    draw.rectangle((0, 0, pil_image.width, height), fill=(255, 255, 255))
    y = 5
    for line in lines:
        draw.text((10, y), line, font=font, fill=(0, 0, 0))
        y += line_height
    image[:, :, :] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def annotate_and_save(
    image_path: Path,
    output_path: Path,
    header_lines: Sequence[str],
    box_label: str,
    detector=None,
) -> int:
    image = imread_unicode(image_path)
    if image is None:
        raise ValueError(f"Unreadable image: {image_path}")

    boxes = predict_boxes(detector, image)
    for box in boxes:
        draw_box(image, box, box_label)
    draw_header(image, header_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite_unicode(output_path, image):
        raise ValueError(f"Failed to write image: {output_path}")
    return len(boxes)


def write_report_csv(rows: Sequence[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "output_path",
        "true_class_id",
        "true_class_cn",
        "pred_class_id",
        "pred_class_cn",
        "confidence",
        "correct",
        "box_count",
        "probabilities",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["probabilities"] = json.dumps(csv_row["probabilities"], ensure_ascii=False)
            writer.writerow(csv_row)


def evaluate_labeled_samples(
    model,
    rows: Sequence[LabelRow],
    image_size: int,
    device: torch.device,
    output_root: Path,
    detector=None,
) -> tuple[int, int]:
    report_rows: list[dict] = []
    correct_count = 0
    for row in rows:
        prediction = predict_image(model, row.image_path, image_size=image_size, device=device)
        pred_id = int(prediction["class_id"])
        pred_cn = str(prediction["class_name_cn"])
        true_cn = CLASS_ID_TO_CN[row.label]
        is_correct = pred_id == row.label
        correct_count += int(is_correct)

        folder = output_root / "labeled_by_truth" / safe_folder_name(row.label, true_cn)
        output_name = f"{row.image_path.stem}__pred_{safe_folder_name(pred_id, pred_cn)}__conf_{prediction['confidence']:.3f}{row.image_path.suffix}"
        output_path = folder / output_name
        box_count = annotate_and_save(
            row.image_path,
            output_path,
            [
                f"真实: {true_cn}",
                f"预测: {pred_cn}  置信度: {prediction['confidence']:.3f}  {'正确' if is_correct else '错误'}",
            ],
            box_label=pred_cn,
            detector=detector,
        )
        report_rows.append(
            {
                "source_path": str(row.image_path),
                "output_path": str(output_path),
                "true_class_id": row.label,
                "true_class_cn": true_cn,
                "pred_class_id": pred_id,
                "pred_class_cn": pred_cn,
                "confidence": prediction["confidence"],
                "correct": is_correct,
                "box_count": box_count,
                "probabilities": prediction["probabilities"],
            }
        )

    write_report_csv(report_rows, output_root / "labeled_report.csv")
    return correct_count, len(rows)


def evaluate_unlabeled_samples(
    model,
    rows: Sequence[TestRow],
    image_size: int,
    device: torch.device,
    output_root: Path,
    detector=None,
) -> int:
    report_rows: list[dict] = []
    for row in rows:
        prediction = predict_image(model, row.image_path, image_size=image_size, device=device)
        pred_id = int(prediction["class_id"])
        pred_cn = str(prediction["class_name_cn"])
        folder = output_root / "unlabeled_by_prediction" / safe_folder_name(pred_id, pred_cn)
        output_name = f"{row.image_path.stem}__pred_{safe_folder_name(pred_id, pred_cn)}__conf_{prediction['confidence']:.3f}{row.image_path.suffix}"
        output_path = folder / output_name
        box_count = annotate_and_save(
            row.image_path,
            output_path,
            [f"预测: {pred_cn}  置信度: {prediction['confidence']:.3f}"],
            box_label=pred_cn,
            detector=detector,
        )
        report_rows.append(
            {
                "source_path": str(row.image_path),
                "output_path": str(output_path),
                "true_class_id": "",
                "true_class_cn": "",
                "pred_class_id": pred_id,
                "pred_class_cn": pred_cn,
                "confidence": prediction["confidence"],
                "correct": "",
                "box_count": box_count,
                "probabilities": prediction["probabilities"],
            }
        )

    write_report_csv(report_rows, output_root / "unlabeled_report.csv")
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained classifier with sampled images and annotated outputs")
    parser.add_argument("--env", choices=["auto", "windows", "linux"], default="auto")
    parser.add_argument("--model", type=Path, default=Path("outputs/classification_results/deploy/classifier.torchscript.pt"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classification_results/evaluation_samples"))
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--yolo-weights", type=Path, default=Path("best.pt"))
    parser.add_argument("--no-yolo", action="store_true", help="Only draw classification headers, no detection boxes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.env)
    if args.data_root is not None:
        config = config.__class__(**{**config.__dict__, "classification_root": args.data_root})

    data_root = resolve_path(config, config.classification_root)
    output_root = resolve_path(config, args.output_dir)
    model_path = resolve_path(config, args.model)
    yolo_weights = None if args.no_yolo else resolve_path(config, args.yolo_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(model_path, device)
    image_size = resolve_inference_image_size(model_path, config.image_size)
    detector = load_yolo_detector(yolo_weights)
    if detector is None:
        print("YOLO weights not found or disabled; output images will contain classification headers without defect boxes.")

    labeled_rows = sample_labeled_rows_by_class(build_label_rows(data_root), per_class=args.per_class, seed=args.seed)
    test_rows = sample_unlabeled_rows(build_test_rows(data_root, config.test_folder_name), count=args.test_count, seed=args.seed)

    correct, labeled_total = evaluate_labeled_samples(model, labeled_rows, image_size, device, output_root, detector=detector)
    unlabeled_total = evaluate_unlabeled_samples(model, test_rows, image_size, device, output_root, detector=detector)

    accuracy = correct / labeled_total * 100.0 if labeled_total else 0.0
    print(f"labeled_samples={labeled_total} correct={correct} accuracy={accuracy:.2f}%")
    print(f"unlabeled_samples={unlabeled_total}")
    print(f"output_dir={output_root}")
    print(f"labeled_report={output_root / 'labeled_report.csv'}")
    print(f"unlabeled_report={output_root / 'unlabeled_report.csv'}")


if __name__ == "__main__":
    main()
