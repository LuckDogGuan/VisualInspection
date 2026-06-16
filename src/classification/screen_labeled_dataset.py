from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Sequence

import torch
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.config import CLASS_ID_TO_CN, get_config, resolve_path
from classification.data import LabelRow, build_label_rows
from classification.inference import load_model_for_inference, resolve_inference_image_size
from classification.train import build_eval_transforms


def safe_folder_name(class_id: int, class_name: str) -> str:
    safe_name = class_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"{class_id:02d}_{safe_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen all labeled classification images with a trained classifier")
    parser.add_argument("--env", choices=["auto", "windows", "linux"], default="auto")
    parser.add_argument("--model", type=Path, default=Path("outputs/classification_results/deploy/classifier.torchscript.pt"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classification_results/full_screening"))
    parser.add_argument("--low-confidence", type=float, default=0.70, help="Images below this confidence are copied for review")
    parser.add_argument("--copy-all", action="store_true", help="Also copy correctly classified images")
    parser.add_argument("--no-copy", action="store_true", help="Only write CSV and summary, do not copy review images")
    return parser.parse_args()


def predict_rows(
    model,
    rows: Sequence[LabelRow],
    image_size: int,
    device: torch.device,
) -> list[dict]:
    transform = build_eval_transforms(image_size)
    records: list[dict] = []

    for index, row in enumerate(rows, start=1):
        image = Image.open(row.image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()

        top_values, top_ids = torch.topk(probabilities, k=min(3, probabilities.numel()))
        pred_id = int(top_ids[0].item())
        confidence = float(top_values[0].item())
        true_cn = CLASS_ID_TO_CN[row.label]
        pred_cn = CLASS_ID_TO_CN[pred_id]

        records.append(
            {
                "source_path": str(row.image_path),
                "true_class_id": row.label,
                "true_class_cn": true_cn,
                "pred_class_id": pred_id,
                "pred_class_cn": pred_cn,
                "confidence": confidence,
                "correct": pred_id == row.label,
                "top3_hit": row.label in {int(item.item()) for item in top_ids},
                "top3_class_ids": "|".join(str(int(item.item())) for item in top_ids),
                "top3_class_cn": "|".join(CLASS_ID_TO_CN[int(item.item())] for item in top_ids),
                "top3_confidence": "|".join(f"{float(item.item()):.6f}" for item in top_values),
            }
        )

        if index % 200 == 0 or index == len(rows):
            print(f"processed={index}/{len(rows)}")

    return records


def write_report_csv(records: Sequence[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "true_class_id",
        "true_class_cn",
        "pred_class_id",
        "pred_class_cn",
        "confidence",
        "correct",
        "top3_hit",
        "top3_class_ids",
        "top3_class_cn",
        "top3_confidence",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def copy_review_images(records: Sequence[dict], output_dir: Path, low_confidence: float, copy_all: bool) -> None:
    for record in records:
        source_path = Path(record["source_path"])
        true_id = int(record["true_class_id"])
        pred_id = int(record["pred_class_id"])
        true_cn = str(record["true_class_cn"])
        pred_cn = str(record["pred_class_cn"])
        confidence = float(record["confidence"])
        is_correct = bool(record["correct"])

        if not is_correct:
            target_root = output_dir / "wrong_by_truth" / safe_folder_name(true_id, true_cn)
        elif confidence < low_confidence:
            target_root = output_dir / "low_confidence" / safe_folder_name(true_id, true_cn)
        elif copy_all:
            target_root = output_dir / "correct_by_truth" / safe_folder_name(true_id, true_cn)
        else:
            continue

        target_root.mkdir(parents=True, exist_ok=True)
        target_name = (
            f"{source_path.stem}"
            f"__true_{safe_folder_name(true_id, true_cn)}"
            f"__pred_{safe_folder_name(pred_id, pred_cn)}"
            f"__conf_{confidence:.3f}"
            f"{source_path.suffix}"
        )
        shutil.copy2(source_path, target_root / target_name)


def write_summary(records: Sequence[dict], output_md: Path, report_csv: Path, low_confidence: float) -> None:
    total = len(records)
    correct = sum(1 for record in records if record["correct"])
    top3_hit = sum(1 for record in records if record["top3_hit"])
    low_count = sum(1 for record in records if float(record["confidence"]) < low_confidence)

    by_true: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_true[int(record["true_class_id"])].append(record)

    mistakes = Counter(
        (str(record["true_class_cn"]), str(record["pred_class_cn"]))
        for record in records
        if not record["correct"]
    )

    lines = [
        "# 全量分类图片筛选结果",
        "",
        f"- 图片数量：{total}",
        f"- Top1 正确：{correct}",
        f"- Top1 准确率：{correct / total * 100:.2f}%" if total else "- Top1 准确率：0.00%",
        f"- Top3 命中：{top3_hit}",
        f"- Top3 命中率：{top3_hit / total * 100:.2f}%" if total else "- Top3 命中率：0.00%",
        f"- 低置信度阈值：{low_confidence:.2f}",
        f"- 低置信度图片：{low_count}",
        f"- 详细 CSV：`{report_csv}`",
        "",
        "## 各类别统计",
        "",
        "| 类别 ID | 类别 | 图片数 | Top1 正确 | Top1 准确率 | Top3 命中率 | 平均置信度 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for class_id in sorted(by_true):
        rows = by_true[class_id]
        row_count = len(rows)
        row_correct = sum(1 for row in rows if row["correct"])
        row_top3 = sum(1 for row in rows if row["top3_hit"])
        avg_conf = mean(float(row["confidence"]) for row in rows)
        lines.append(
            f"| {class_id} | {CLASS_ID_TO_CN[class_id]} | {row_count} | {row_correct} | "
            f"{row_correct / row_count * 100:.2f}% | {row_top3 / row_count * 100:.2f}% | {avg_conf:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 主要错分方向",
            "",
            "| 真实类别 | 预测类别 | 数量 |",
            "| --- | --- | ---: |",
        ]
    )
    for (true_cn, pred_cn), count in mistakes.most_common(30):
        lines.append(f"| {true_cn} | {pred_cn} | {count} |")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = get_config(args.env)
    if args.data_root is not None:
        config = config.__class__(**{**config.__dict__, "classification_root": args.data_root})

    data_root = resolve_path(config, config.classification_root)
    model_path = resolve_path(config, args.model)
    output_dir = resolve_path(config, args.output_dir)
    report_csv = output_dir / "all_labeled_predictions.csv"
    summary_md = output_dir / "summary.md"

    rows = build_label_rows(data_root)
    if not rows:
        raise RuntimeError(f"No labeled images found: {data_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(model_path, device)
    image_size = resolve_inference_image_size(model_path, config.image_size)

    records = predict_rows(model, rows, image_size=image_size, device=device)
    write_report_csv(records, report_csv)
    write_summary(records, summary_md, report_csv, args.low_confidence)

    if not args.no_copy:
        copy_review_images(records, output_dir, low_confidence=args.low_confidence, copy_all=args.copy_all)

    correct = sum(1 for record in records if record["correct"])
    top3_hit = sum(1 for record in records if record["top3_hit"])
    low_count = sum(1 for record in records if float(record["confidence"]) < args.low_confidence)
    print(f"images={len(records)}")
    print(f"top1_correct={correct} top1_accuracy={correct / len(records) * 100:.2f}%")
    print(f"top3_hit={top3_hit} top3_accuracy={top3_hit / len(records) * 100:.2f}%")
    print(f"low_confidence_lt_{args.low_confidence:.2f}={low_count}")
    print(f"csv={report_csv}")
    print(f"summary={summary_md}")
    if not args.no_copy:
        print(f"wrong_images={output_dir / 'wrong_by_truth'}")
        print(f"low_confidence_images={output_dir / 'low_confidence'}")


if __name__ == "__main__":
    main()
