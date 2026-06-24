from __future__ import annotations

import argparse
import csv
import html
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_yolo_annotation_review_guides import CLASS_ORDER, YoloBox, draw_text_box, load_font


@dataclass(frozen=True)
class PredBox:
    box: YoloBox
    conf: float | None = None


@dataclass(frozen=True)
class ReviewItem:
    image_name: str
    class_name: str
    score: int
    issue: str
    gt_count: int
    pred_count: int
    matched_count: int
    max_iou: float
    guide_image: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create YOLO GT-vs-prediction review guides.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/yolo_stage3_manual/exports/dataset_clean_candidate_20260623_v3"),
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("server_outputs/predict_clean_candidate_20260623_v3_val"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/yolo_stage3_manual/review/v3_prediction_review"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-items", type=int, default=40)
    parser.add_argument("--iou-match", type=float, default=0.35)
    return parser.parse_args()


def read_label(path: Path, with_conf: bool) -> list[PredBox]:
    if not path.exists():
        return []
    boxes: list[PredBox] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            box = YoloBox(int(parts[0]), *(float(value) for value in parts[1:5]))
            conf = float(parts[5]) if with_conf and len(parts) >= 6 else None
        except ValueError:
            continue
        boxes.append(PredBox(box=box, conf=conf))
    return boxes


def box_iou(left: YoloBox, right: YoloBox) -> float:
    lx1, ly1 = left.cx - left.width / 2, left.cy - left.height / 2
    lx2, ly2 = left.cx + left.width / 2, left.cy + left.height / 2
    rx1, ry1 = right.cx - right.width / 2, right.cy - right.height / 2
    rx2, ry2 = right.cx + right.width / 2, right.cy + right.height / 2
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0


def analyze_pair(
    image_name: str,
    gt_boxes: list[PredBox],
    pred_boxes: list[PredBox],
    iou_match: float,
) -> tuple[int, str, int, float]:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    max_iou = 0.0
    class_mismatch = 0

    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(gt_boxes):
        for pred_index, pred in enumerate(pred_boxes):
            iou = box_iou(gt.box, pred.box)
            max_iou = max(max_iou, iou)
            if iou >= iou_match:
                candidates.append((iou, gt_index, pred_index))
    candidates.sort(reverse=True)

    for iou, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        if gt_boxes[gt_index].box.class_id == pred_boxes[pred_index].box.class_id:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
        else:
            class_mismatch += 1
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)

    missing = len(gt_boxes) - len(matched_gt)
    extra = len(pred_boxes) - len(matched_pred)
    duplicate = 0
    for pred_index, pred in enumerate(pred_boxes):
        if pred_index in matched_pred:
            continue
        if any(
            gt.box.class_id == pred.box.class_id and box_iou(gt.box, pred.box) >= 0.15
            for gt in gt_boxes
        ):
            duplicate += 1

    issues: list[str] = []
    score = 0
    if missing:
        issues.append(f"漏检 {missing}")
        score += missing * 5
    if class_mismatch:
        issues.append(f"类别疑似错误 {class_mismatch}")
        score += class_mismatch * 4
    if duplicate:
        issues.append(f"重复框 {duplicate}")
        score += duplicate * 3
    remaining_extra = max(0, extra - duplicate)
    if remaining_extra:
        issues.append(f"多余框 {remaining_extra}")
        score += remaining_extra * 2
    if gt_boxes and pred_boxes and max_iou < 0.35:
        issues.append("框偏差大")
        score += 3
    if not issues:
        issues.append("基本可用")
    return score, "；".join(issues), len(matched_gt) - class_mismatch, max_iou


def draw_boxes(
    image: Image.Image,
    boxes: list[PredBox],
    color: tuple[int, int, int],
    label_prefix: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    for index, pred in enumerate(boxes, 1):
        x1, y1, x2, y2 = pred.box.xyxy(width, height)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        class_name = CLASS_ORDER[pred.box.class_id] if pred.box.class_id < len(CLASS_ORDER) else str(pred.box.class_id)
        conf = f" {pred.conf:.2f}" if pred.conf is not None else ""
        draw_text_box(draw, (x1, max(0, y1 - 24)), f"{label_prefix}{index} {class_name}{conf}", font, color)
    return output


def resize(image: Image.Image, max_width: int = 620) -> Image.Image:
    if image.width <= max_width:
        return image.copy()
    ratio = max_width / image.width
    return image.resize((max_width, max(1, round(image.height * ratio))), Image.Resampling.LANCZOS)


def make_guide_image(
    source_image: Path,
    gt_boxes: list[PredBox],
    pred_boxes: list[PredBox],
    output_path: Path,
    title: str,
) -> None:
    font = load_font(20)
    small_font = load_font(16)
    original = Image.open(source_image).convert("RGB")
    left = draw_boxes(resize(original), gt_boxes, (20, 170, 80), "GT", font)
    right = draw_boxes(resize(original), pred_boxes, (40, 90, 220), "P", font)
    header_h = 42
    gap = 16
    canvas = Image.new("RGB", (left.width + right.width + gap, max(left.height, right.height) + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, font=small_font, fill=(20, 20, 20))
    canvas.paste(left, (0, header_h))
    canvas.paste(right, (left.width + gap, header_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def class_from_gt(gt_boxes: list[PredBox], image_name: str) -> str:
    if gt_boxes:
        class_id = gt_boxes[0].box.class_id
        if 0 <= class_id < len(CLASS_ORDER):
            return CLASS_ORDER[class_id]
    return image_name.split("_batch_", 1)[0]


def build_review(
    dataset_root: Path,
    prediction_root: Path,
    output_dir: Path,
    split: str,
    max_items: int,
    iou_match: float,
) -> list[ReviewItem]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    pred_label_dir = prediction_root / "labels"
    guide_dir = output_dir / "guides"
    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[ReviewItem] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        gt_boxes = read_label(label_dir / f"{image_path.stem}.txt", with_conf=False)
        pred_boxes = read_label(pred_label_dir / f"{image_path.stem}.txt", with_conf=True)
        score, issue, matched, max_iou = analyze_pair(image_path.name, gt_boxes, pred_boxes, iou_match)
        guide_name = f"{image_path.stem}_review.jpg"
        make_guide_image(
            image_path,
            gt_boxes,
            pred_boxes,
            guide_dir / guide_name,
            f"{image_path.name} | {issue}",
        )
        items.append(
            ReviewItem(
                image_name=image_path.name,
                class_name=class_from_gt(gt_boxes, image_path.name),
                score=score,
                issue=issue,
                gt_count=len(gt_boxes),
                pred_count=len(pred_boxes),
                matched_count=matched,
                max_iou=max_iou,
                guide_image=f"guides/{guide_name}",
            )
        )

    items.sort(key=lambda item: (item.score, item.class_name, item.image_name), reverse=True)
    write_outputs(output_dir, items[:max_items], items)
    return items


def write_outputs(output_dir: Path, top_items: list[ReviewItem], all_items: list[ReviewItem]) -> None:
    fields = [
        "image_name",
        "class_name",
        "score",
        "issue",
        "gt_count",
        "pred_count",
        "matched_count",
        "max_iou",
        "guide_image",
    ]
    for name, rows in [("review_priority.csv", top_items), ("review_all.csv", all_items)]:
        with (output_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in rows:
                writer.writerow({field: getattr(item, field) for field in fields})

    rows_html = []
    for item in top_items:
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(item.score))}</td>"
            f"<td>{html.escape(item.class_name)}</td>"
            f"<td>{html.escape(item.issue)}</td>"
            f"<td>{html.escape(item.image_name)}</td>"
            f"<td>GT {item.gt_count} / P {item.pred_count} / match {item.matched_count}</td>"
            f"<td>{item.max_iou:.3f}</td>"
            f"<td><img src=\"{html.escape(item.guide_image)}\" loading=\"lazy\"></td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>v3 预测复核清单</title>
<style>
body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 24px; color: #172033; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #d8dee9; padding: 8px; vertical-align: top; text-align: left; }}
th {{ position: sticky; top: 0; background: #fff; }}
img {{ max-width: 980px; width: 100%; height: auto; border: 1px solid #ccd3df; }}
.note {{ margin: 8px 0 18px; color: #4a5568; }}
</style>
</head>
<body>
<h1>v3 预测复核清单</h1>
<p class="note">左侧绿色是人工标注 GT，右侧蓝色是 v3 best 模型预测。优先看 score 高的样本。</p>
<table>
<thead><tr><th>score</th><th>类别</th><th>问题</th><th>图片</th><th>数量</th><th>max IoU</th><th>对照图</th></tr></thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    items = build_review(
        dataset_root=args.dataset_root,
        prediction_root=args.prediction_root,
        output_dir=args.output_dir,
        split=args.split,
        max_items=args.max_items,
        iou_match=args.iou_match,
    )
    print(f"review items: {len(items)}")
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
