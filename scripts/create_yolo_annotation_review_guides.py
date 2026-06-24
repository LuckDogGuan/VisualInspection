from __future__ import annotations

import argparse
import csv
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASS_ORDER = ["dent", "powder", "stain", "crack", "transverse_bump"]
CLASS_CN = {
    "dent": "碰伤",
    "powder": "凸粉",
    "stain": "脏点",
    "crack": "涂层开裂",
    "transverse_bump": "横条压凹",
}
DEFAULT_CLASSES = ["stain", "dent", "powder"]
DEFAULT_BATCHES = ["batch_004", "batch_005"]


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float

    def xyxy(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        x1 = (self.cx - self.width / 2) * image_width
        y1 = (self.cy - self.height / 2) * image_height
        x2 = (self.cx + self.width / 2) * image_width
        y2 = (self.cy + self.height / 2) * image_height
        return (
            max(0, min(image_width - 1, round(x1))),
            max(0, min(image_height - 1, round(y1))),
            max(0, min(image_width - 1, round(x2))),
            max(0, min(image_height - 1, round(y2))),
        )

    @property
    def area(self) -> float:
        return self.width * self.height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create side-by-side YOLO annotation review images with Chinese filename support."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--batches", nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--decisions", type=Path, default=None)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def read_registry(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def path_key(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/").lower()


def read_decisions(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        path_key(row["batch_label_path"]): row
        for row in rows
        if row.get("batch_label_path") and row.get("decision")
    }


def read_yolo_boxes(path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            boxes.append(YoloBox(int(parts[0]), *(float(value) for value in parts[1:])))
        except ValueError:
            continue
    return boxes


def natural_priority(row: dict[str, str]) -> tuple[int, str]:
    label_path = Path(row["batch_label_path"])
    boxes = read_yolo_boxes(label_path)
    # Dense annotations and many tiny boxes are most useful for this review.
    tiny_count = sum(1 for box in boxes if box.area < 0.0025)
    return (len(boxes) + tiny_count, Path(row["batch_image_path"]).name)


def select_rows(
    rows: list[dict[str, str]],
    classes: list[str],
    batches: list[str],
    per_class: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for class_name in classes:
        candidates = [
            row
            for row in rows
            if row.get("status") == "annotated"
            and row.get("class") == class_name
            and row.get("batch_id") in batches
            and Path(row.get("batch_image_path", "")).exists()
            and Path(row.get("batch_label_path", "")).exists()
        ]
        candidates.sort(key=natural_priority, reverse=True)
        selected.extend(candidates[:per_class])
    return selected


def resize_for_review(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image.copy()
    ratio = max_width / image.width
    new_size = (max_width, max(1, round(image.height * ratio)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=(255, 255, 255))
    draw.text((x, y), text, font=font, fill=fill)


def draw_boxes(
    image: Image.Image,
    boxes: list[YoloBox],
    color: tuple[int, int, int],
    label_prefix: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    for index, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = box.xyxy(width, height)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        if index <= 8:
            draw_text_box(draw, (x1, max(0, y1 - 22)), f"{label_prefix}{index}", font, color)
    return output


def union_box(boxes: list[YoloBox]) -> YoloBox | None:
    if not boxes:
        return None
    x1 = min(box.cx - box.width / 2 for box in boxes)
    y1 = min(box.cy - box.height / 2 for box in boxes)
    x2 = max(box.cx + box.width / 2 for box in boxes)
    y2 = max(box.cy + box.height / 2 for box in boxes)
    margin = 0.015
    x1 = max(0.0, x1 - margin)
    y1 = max(0.0, y1 - margin)
    x2 = min(1.0, x2 + margin)
    y2 = min(1.0, y2 + margin)
    return YoloBox(boxes[0].class_id, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def boxes_overlap_or_close(left: YoloBox, right: YoloBox, gap: float) -> bool:
    left_x1 = left.cx - left.width / 2 - gap
    left_y1 = left.cy - left.height / 2 - gap
    left_x2 = left.cx + left.width / 2 + gap
    left_y2 = left.cy + left.height / 2 + gap
    right_x1 = right.cx - right.width / 2
    right_y1 = right.cy - right.height / 2
    right_x2 = right.cx + right.width / 2
    right_y2 = right.cy + right.height / 2
    return not (left_x2 < right_x1 or right_x2 < left_x1 or left_y2 < right_y1 or right_y2 < left_y1)


def cluster_boxes(boxes: list[YoloBox], gap: float = 0.035) -> list[list[YoloBox]]:
    clusters: list[list[YoloBox]] = []
    for box in boxes:
        matched_indexes = [
            index
            for index, cluster in enumerate(clusters)
            if any(boxes_overlap_or_close(box, member, gap) for member in cluster)
        ]
        if not matched_indexes:
            clusters.append([box])
            continue
        first = matched_indexes[0]
        clusters[first].append(box)
        for index in reversed(matched_indexes[1:]):
            clusters[first].extend(clusters.pop(index))
    return clusters


def merge_local_clusters(boxes: list[YoloBox], min_cluster_size: int = 3) -> list[YoloBox]:
    suggested: list[YoloBox] = []
    for cluster in cluster_boxes(boxes):
        if len(cluster) >= min_cluster_size:
            merged = union_box(cluster)
            if merged is not None:
                suggested.append(merged)
        else:
            suggested.extend(cluster)
    suggested.sort(key=lambda box: (box.cy, box.cx))
    return suggested


def expand_box(box: YoloBox, factor: float) -> YoloBox:
    width = min(1.0, box.width * factor)
    height = min(1.0, box.height * factor)
    x1 = max(0.0, box.cx - width / 2)
    y1 = max(0.0, box.cy - height / 2)
    x2 = min(1.0, box.cx + width / 2)
    y2 = min(1.0, box.cy + height / 2)
    return YoloBox(box.class_id, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def parse_indices(text: str) -> list[int]:
    return [int(item.strip()) for item in text.replace("+", ",").split(",") if item.strip()]


def apply_manual_decision(
    old_boxes: list[YoloBox],
    suggested_boxes: list[YoloBox],
    decision: str,
) -> tuple[list[YoloBox], str, str]:
    decision = decision.strip()
    if not decision:
        return suggested_boxes, "", ""
    if decision == "skip":
        return [], "manual:skip", "人工复核：跳过该图，不进入 clean dataset。"

    if decision.startswith("keep_old:"):
        indexes = parse_indices(decision.split(":", 1)[1])
        boxes = [old_boxes[index - 1] for index in indexes if 1 <= index <= len(old_boxes)]
        return boxes, f"manual:{decision}", "人工复核：只保留指定 old 框。"

    remove_new: set[int] = set()
    additions: list[YoloBox] = []
    replacements: dict[int, YoloBox] = {}
    notes: list[str] = []

    for operation in [item.strip() for item in decision.split(";") if item.strip()]:
        if operation.startswith("delete_new:"):
            remove_new.update(parse_indices(operation.split(":", 1)[1]))
            notes.append("删除指定 new 框")
        elif operation.startswith("merge_new:"):
            indexes = parse_indices(operation.split(":", 1)[1])
            boxes = [suggested_boxes[index - 1] for index in indexes if 1 <= index <= len(suggested_boxes)]
            merged = union_box(boxes)
            if merged is not None:
                remove_new.update(indexes)
                additions.append(merged)
                notes.append("合并指定 new 框")
        elif operation.startswith("expand_new:"):
            _, index_text, factor_text = operation.split(":", 2)
            index = int(index_text)
            if 1 <= index <= len(suggested_boxes):
                replacements[index] = expand_box(suggested_boxes[index - 1], float(factor_text))
                notes.append("放大指定 new 框")
        elif operation.startswith("split_new_to_old:"):
            left, right = operation.split("=", 1)
            new_index = int(left.split(":", 1)[1])
            old_indexes = parse_indices(right)
            remove_new.add(new_index)
            additions.extend(old_boxes[index - 1] for index in old_indexes if 1 <= index <= len(old_boxes))
            notes.append("把指定 new 框拆回 old 框")
        else:
            raise ValueError(f"Unsupported manual decision operation: {operation}")

    output: list[YoloBox] = []
    for index, box in enumerate(suggested_boxes, 1):
        if index in remove_new:
            continue
        output.append(replacements.get(index, box))
    output.extend(additions)
    output.sort(key=lambda box: (box.cy, box.cx))
    reason = "人工复核：" + "、".join(notes) if notes else "人工复核调整。"
    return output, f"manual:{decision}", reason


def suggest_boxes(class_name: str, boxes: list[YoloBox]) -> tuple[list[YoloBox], str, str]:
    if not boxes:
        return [], "skip", "没有有效标签，建议跳过或重新确认。"

    tiny_count = sum(1 for box in boxes if box.area < 0.0025)
    total_area = sum(box.area for box in boxes)

    if class_name in {"stain", "dent", "powder"} and len(boxes) >= 3:
        suggested = merge_local_clusters(boxes)
        return suggested, "merge", "相邻小框按局部合并；分散独立缺陷保留分开，避免一个大框包住过多正常区域。"
    if class_name == "stain" and tiny_count >= 3 and total_area < 0.012:
        return boxes, "skip?", "多个极小点需要人工确认；如果肉眼不稳定，建议跳过，不进 clean dataset。"
    if class_name == "dent" and len(boxes) >= 3:
        return merge_local_clusters(boxes, min_cluster_size=2), "merge", "同一条连续碰伤不要切成很多小框，建议按局部损伤合并。"
    if class_name == "powder" and len(boxes) >= 3:
        return merge_local_clusters(boxes, min_cluster_size=2), "merge", "同一局部多个凸粉点可先合并成粗区域框，避免框选尺度过碎。"
    return boxes, "keep", "当前框数量和尺度基本可用于第一轮粗定位复核。"


def truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = max(4, (limit - 3) // 2)
    return f"{text[:keep]}...{text[-keep:]}"


def draw_header(
    image: Image.Image,
    title: str,
    subtitle: str,
    font_title: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_body: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    header_height = 104
    output = Image.new("RGB", (image.width, image.height + header_height), (248, 250, 252))
    output.paste(image, (0, header_height))
    draw = ImageDraw.Draw(output)
    draw.text((14, 12), title, font=font_title, fill=(15, 23, 42))
    subtitle_lines = textwrap.wrap(subtitle, width=38)
    for index, line in enumerate(subtitle_lines[:2]):
        draw.text((14, 48 + index * 24), line, font=font_body, fill=(71, 85, 105))
    return output


def make_review_image(
    row: dict[str, str],
    index: int,
    output_dir: Path,
    max_width: int,
    font_title: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_body: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_small: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    decisions: dict[str, dict[str, str]],
) -> dict[str, str]:
    image_path = Path(row["batch_image_path"])
    label_path = Path(row["batch_label_path"])
    class_name = row["class"]
    boxes = read_yolo_boxes(label_path)
    suggested, action, reason = suggest_boxes(class_name, boxes)
    decision_row = decisions.get(path_key(label_path))
    if decision_row is not None:
        suggested, action, reason = apply_manual_decision(boxes, suggested, decision_row["decision"])
        if decision_row.get("notes"):
            reason = f"{reason} {decision_row['notes']}"

    image = Image.open(image_path).convert("RGB")
    image = resize_for_review(image, max_width)
    left = draw_boxes(image, boxes, (220, 38, 38), "old-", font_small)
    right = draw_boxes(image, suggested, (16, 185, 129), "new-", font_small)

    class_cn = CLASS_CN.get(class_name, row.get("true_class_cn") or class_name)
    file_display = truncate_middle(image_path.name, 42)
    left = draw_header(left, "当前标注", f"{class_name}/{class_cn}  boxes={len(boxes)}", font_title, font_body)
    action_title = "人工调整" if action.startswith("manual:") else f"建议：{action}"
    right = draw_header(right, action_title, reason, font_title, font_body)

    gap = 18
    note_height = 120
    canvas_width = left.width + right.width + gap
    canvas_height = max(left.height, right.height) + note_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (241, 245, 249))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))

    draw = ImageDraw.Draw(canvas)
    y = max(left.height, right.height) + 16
    draw.text((14, y), f"样例 {index:03d} | {row['batch_id']} | {file_display}", font=font_body, fill=(15, 23, 42))
    wrapped_reason = textwrap.wrap(f"调整说明：{reason}", width=64)
    for offset, line in enumerate(wrapped_reason[:2], 1):
        draw.text((14, y + offset * 28), line, font=font_body, fill=(51, 65, 85))

    output_name = f"review_{index:03d}_{class_name}_{row['batch_id']}.jpg"
    output_path = output_dir / output_name
    canvas.save(output_path, quality=92)
    return {
        "review_id": f"review_{index:03d}",
        "class": class_name,
        "class_cn": class_cn,
        "batch_id": row["batch_id"],
        "action": action,
        "reason": reason,
        "manual_decision": decision_row["decision"] if decision_row is not None else "",
        "box_count_old": str(len(boxes)),
        "box_count_suggested": str(len(suggested)),
        "review_image": str(output_path),
        "batch_image_path": row["batch_image_path"],
        "batch_label_path": row["batch_label_path"],
        "source_path": row.get("source_path", ""),
        "original_filename": image_path.name,
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    cards = []
    for row in rows:
        image_name = Path(row["review_image"]).name
        cards.append(
            "\n".join(
                [
                    '<article class="card">',
                    f'  <img src="{image_name}" alt="{row["review_id"]}">',
                    "  <div class=\"meta\">",
                    f'    <b>{row["review_id"]} | {row["class"]}/{row["class_cn"]} | {row["action"]}</b>',
                    f'    <span>{row["original_filename"]}</span>',
                    "  </div>",
                    "</article>",
                ]
            )
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>YOLO 标注前后对照复核</title>
  <style>
    body {{ margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; background: #f4f6f8; color: #17202a; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p {{ margin: 0 0 22px; color: #536471; line-height: 1.6; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 22px; }}
    .card {{ background: #fff; border: 1px solid #d8dee6; border-radius: 8px; overflow: hidden; }}
    .card img {{ display: block; width: 100%; }}
    .meta {{ display: flex; gap: 16px; align-items: center; padding: 10px 12px; border-top: 1px solid #e5eaf0; }}
    .meta span {{ color: #64748b; }}
  </style>
</head>
<body>
  <main>
    <h1>YOLO 标注前后对照复核</h1>
    <p>左侧为当前标注框，右侧为建议复核方向。绿色框是辅助建议，不会自动改标签；最终是否 keep / redraw / merge / skip 由人工确认。</p>
    <section class="grid">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    stage_root = project_root / "data" / "yolo_stage3_manual"
    registry_path = stage_root / "annotation_registry.csv"
    output_dir = args.output_dir or stage_root / "review" / "before_after_guides"
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = args.decisions or stage_root / "review" / "manual_review_decisions.csv"
    decisions = read_decisions(decisions_path)

    rows = read_registry(registry_path)
    selected = select_rows(rows, args.classes, args.batches, args.per_class)
    if not selected:
        raise SystemExit("No matching annotated rows found for review guide generation.")

    font_title = load_font(26)
    font_body = load_font(20)
    font_small = load_font(16)
    manifest_rows = [
        make_review_image(row, index, output_dir, args.max_width, font_title, font_body, font_small, decisions)
        for index, row in enumerate(selected, 1)
    ]
    manifest_path = output_dir / "review_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    index_path = output_dir / "index.html"
    write_index(index_path, manifest_rows)
    print(f"output_dir={output_dir}")
    print(f"review_images={len(manifest_rows)}")
    print(f"manifest={manifest_path}")
    print(f"index={index_path}")


if __name__ == "__main__":
    main()
