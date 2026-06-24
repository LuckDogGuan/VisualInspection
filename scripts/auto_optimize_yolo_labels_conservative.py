from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_yolo_annotation_review_guides import CLASS_ORDER, YoloBox, load_font


@dataclass(frozen=True)
class OptimizationRecord:
    split: str
    class_name: str
    image_name: str
    action: str
    reason: str
    before_count: int
    after_count: int
    image_path: str
    label_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservatively optimize YOLO labels from an existing export.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/yolo_stage3_manual/exports/dataset_clean_candidate_20260623_v3"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/yolo_stage3_manual/exports/dataset_clean_candidate_20260624_v4_auto_conservative"),
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("data/yolo_stage3_manual/review/v4_auto_conservative_review"),
    )
    return parser.parse_args()


def read_yolo_label(path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            boxes.append(YoloBox(int(parts[0]), *(float(value) for value in parts[1:])))
        except ValueError:
            continue
    return boxes


def write_yolo_label(path: Path, boxes: list[YoloBox]) -> None:
    lines = [f"{box.class_id} {box.cx:.6f} {box.cy:.6f} {box.width:.6f} {box.height:.6f}" for box in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def union_box(boxes: list[YoloBox], margin: float = 0.01) -> YoloBox:
    x1 = max(0.0, min(box.cx - box.width / 2 for box in boxes) - margin)
    y1 = max(0.0, min(box.cy - box.height / 2 for box in boxes) - margin)
    x2 = min(1.0, max(box.cx + box.width / 2 for box in boxes) + margin)
    y2 = min(1.0, max(box.cy + box.height / 2 for box in boxes) + margin)
    return YoloBox(boxes[0].class_id, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def overlap_or_close(left: YoloBox, right: YoloBox, gap: float) -> bool:
    left_x1 = left.cx - left.width / 2 - gap
    left_y1 = left.cy - left.height / 2 - gap
    left_x2 = left.cx + left.width / 2 + gap
    left_y2 = left.cy + left.height / 2 + gap
    right_x1 = right.cx - right.width / 2
    right_y1 = right.cy - right.height / 2
    right_x2 = right.cx + right.width / 2
    right_y2 = right.cy + right.height / 2
    return not (left_x2 < right_x1 or right_x2 < left_x1 or left_y2 < right_y1 or right_y2 < left_y1)


def cluster_boxes(boxes: list[YoloBox], gap: float) -> list[list[YoloBox]]:
    clusters: list[list[YoloBox]] = []
    for box in boxes:
        matches = [idx for idx, cluster in enumerate(clusters) if any(overlap_or_close(box, other, gap) for other in cluster)]
        if not matches:
            clusters.append([box])
            continue
        first = matches[0]
        clusters[first].append(box)
        for idx in reversed(matches[1:]):
            clusters[first].extend(clusters.pop(idx))
    return clusters


def optimize_stain(boxes: list[YoloBox]) -> tuple[list[YoloBox], str, str]:
    if len(boxes) < 4:
        return boxes, "kept_original", "stain 框数量不多，保守保留。"
    clusters = cluster_boxes(boxes, gap=0.05)
    optimized: list[YoloBox] = []
    changed = False
    for cluster in clusters:
        if len(cluster) >= 3:
            merged = union_box(cluster, margin=0.012)
            # Avoid a single box swallowing normal surface. Visual review showed broad stain unions are unsafe.
            if merged.area <= 0.04 and merged.width <= 0.32 and merged.height <= 0.18:
                optimized.append(merged)
                changed = True
            else:
                optimized.extend(cluster)
        else:
            optimized.extend(cluster)
    if changed and len(optimized) < len(boxes):
        optimized.sort(key=lambda box: (box.cy, box.cx))
        return optimized, "stain_merge_dense_regions", "连续/密集 stain 小框合并成稳定区域框；过大区域未合并。"
    return boxes, "kept_original", "stain 不满足保守合并条件，原样保留。"


def optimize_dent(boxes: list[YoloBox]) -> tuple[list[YoloBox], str, str]:
    clusters = cluster_boxes(boxes, gap=0.018)
    optimized: list[YoloBox] = []
    changed = False
    for cluster in clusters:
        if len(cluster) >= 2:
            merged = union_box(cluster, margin=0.006)
            if merged.area <= 0.045:
                optimized.append(merged)
                changed = True
            else:
                optimized.extend(cluster)
        else:
            optimized.extend(cluster)
    if changed and len(optimized) < len(boxes):
        optimized.sort(key=lambda box: (box.cy, box.cx))
        return optimized, "dent_merge_duplicate_boxes", "同一处 dent 的重叠/极近小框合并；分散缺陷保留分开。"
    return boxes, "kept_original", "dent 未发现明确重复小框，原样保留。"


def optimize_crack(boxes: list[YoloBox]) -> tuple[list[YoloBox], str, str]:
    optimized: list[YoloBox] = []
    changed = False
    for box in boxes:
        if box.width >= 0.55 and box.height >= 0.12:
            optimized.append(YoloBox(box.class_id, box.cx, box.cy, box.width, min(box.height, 0.085)))
            changed = True
        else:
            optimized.append(box)
    if changed:
        return optimized, "crack_shrink_tall_wide_boxes", "很宽且明显过高的 crack 框上下收紧；窄框和不确定框保留。"
    return boxes, "kept_original", "crack 未触发保守收紧条件，原样保留。"


def optimize_boxes_for_class(class_name: str, boxes: list[YoloBox]) -> tuple[list[YoloBox], str, str]:
    if not boxes:
        return boxes, "empty", "无标注框。"
    if class_name == "stain":
        return optimize_stain(boxes)
    if class_name == "dent":
        return optimize_dent(boxes)
    if class_name == "crack":
        return optimize_crack(boxes)
    return boxes, "kept_original", f"{class_name} 本轮不做自动优化。"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_data_yaml(output_root: Path) -> None:
    names = "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_ORDER))
    text = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names}"
    )
    (output_root / "data.yaml").write_text(text, encoding="utf-8")


def draw_boxes(image: Image.Image, boxes: list[YoloBox], color: tuple[int, int, int], prefix: str) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = load_font(18)
    width, height = output.size
    for index, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = box.xyxy(width, height)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1, max(0, y1 - 20)), f"{prefix}{index}", font=font, fill=color)
    return output


def make_before_after(image_path: Path, before: list[YoloBox], after: list[YoloBox], output_path: Path, title: str) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((620, 465), Image.Resampling.LANCZOS)
        left = draw_boxes(image, before, (40, 160, 70), "old")
        right = draw_boxes(image, after, (40, 90, 220), "new")
    header = 34
    gap = 16
    canvas = Image.new("RGB", (left.width + right.width + gap, max(left.height, right.height) + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(20, 20, 20), font=load_font(16))
    canvas.paste(left, (0, header))
    canvas.paste(right, (left.width + gap, header))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def build_auto_optimized_dataset(source_root: Path, output_root: Path, review_dir: Path) -> list[OptimizationRecord]:
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_root}")
    manifest_rows = read_manifest(source_root / "clean_manifest.csv")
    output_rows: list[dict[str, str]] = []
    records: list[OptimizationRecord] = []

    for row in manifest_rows:
        split = row["split"]
        src_image = Path(row["exported_image"])
        src_label = Path(row["exported_label"])
        if not src_image.exists():
            src_image = source_root / "images" / split / src_image.name
        if not src_label.exists():
            src_label = source_root / "labels" / split / src_label.name

        dst_image = output_root / "images" / split / src_image.name
        dst_label = output_root / "labels" / split / src_label.name
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_image, dst_image)

        before = read_yolo_label(src_label)
        after, action, reason = optimize_boxes_for_class(row["class"], before)
        write_yolo_label(dst_label, after)

        new_row = dict(row)
        new_row["action"] = action if action != "kept_original" else row.get("action", action)
        new_row["reason"] = reason
        new_row["box_count_clean"] = str(len(after))
        new_row["exported_image"] = str(dst_image)
        new_row["exported_label"] = str(dst_label)
        output_rows.append(new_row)

        if action != "kept_original":
            review_image = review_dir / "guides" / f"{src_image.stem}_{action}.jpg"
            make_before_after(src_image, before, after, review_image, f"{src_image.name} | {action}")
        records.append(
            OptimizationRecord(
                split=split,
                class_name=row["class"],
                image_name=src_image.name,
                action=action,
                reason=reason,
                before_count=len(before),
                after_count=len(after),
                image_path=str(dst_image),
                label_path=str(dst_label),
            )
        )

    for split in ["train", "val", "test"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    write_data_yaml(output_root)
    write_manifest(output_root / "clean_manifest.csv", output_rows)
    write_review(review_dir, records, output_root)
    return records


def write_review(review_dir: Path, records: list[OptimizationRecord], output_root: Path) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "class_name",
        "image_name",
        "action",
        "reason",
        "before_count",
        "after_count",
        "image_path",
        "label_path",
    ]
    with (review_dir / "optimization_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field) for field in fields})

    changed = [record for record in records if record.action != "kept_original"]
    by_action: dict[str, int] = {}
    for record in changed:
        by_action[record.action] = by_action.get(record.action, 0) + 1
    action_lines = "\n".join(f"- {action}: {count}" for action, count in sorted(by_action.items()))
    sample_lines = "\n".join(
        f"- `{record.image_name}`: {record.action}, {record.before_count} -> {record.after_count}"
        for record in changed[:30]
    )
    readme = f"""# v4 自动保守标注优化

生成时间：2026-06-24

## 结论

- 输出数据集：`{output_root.as_posix()}`
- 总样本：{len(records)}
- 自动修改样本：{len(changed)}

## 修改类型

{action_lines if action_lines else "- 无自动修改"}

## 保守规则

1. `stain`：只合并连续/密集的小框；合并后区域过大则跳过。
2. `dent`：只合并重叠或极近的小框；分散缺陷保留分开。
3. `crack`：只收紧很宽且明显过高的大框；不确定框原样保留。
4. `powder`、`transverse_bump`：本轮不自动改，避免扩展不稳定类别。

## 已修改样本示例

{sample_lines if sample_lines else "- 无"}

## 文件

- `optimization_manifest.csv`：所有样本的动作记录。
- `guides/`：有修改样本的 old/new 对照图。
"""
    (review_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = build_auto_optimized_dataset(args.source_root, args.output_root, args.review_dir)
    changed = sum(1 for record in records if record.action != "kept_original")
    print(f"records: {len(records)}")
    print(f"changed: {changed}")
    print(args.output_root)
    print(args.review_dir / "README.md")


if __name__ == "__main__":
    main()
