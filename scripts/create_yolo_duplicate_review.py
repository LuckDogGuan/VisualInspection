from __future__ import annotations

import argparse
import csv
import html
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_yolo_annotation_review_guides import CLASS_ORDER


@dataclass(frozen=True)
class DuplicateCandidate:
    row_index: int
    image_name: str
    class_name: str
    batch_id: str
    source_key: str
    hash_value: str
    action: str
    image_path: str
    label_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find near-duplicate YOLO samples and export a dedup candidate.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/yolo_stage3_manual/exports/dataset_clean_candidate_20260623_v3"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/yolo_stage3_manual/exports/dataset_clean_candidate_20260624_v4_dedup"),
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("data/yolo_stage3_manual/review/v4_duplicate_review"),
    )
    parser.add_argument("--hash-threshold", type=int, default=4)
    return parser.parse_args()


def normalize_source_key(row: dict[str, str]) -> str:
    source = Path(row.get("source_path") or row.get("batch_image_path") or row.get("original_filename") or "")
    stem = source.stem.lower()
    for prefix in [f"{name}__" for name in CLASS_ORDER]:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    return f"{row.get('class', '')}:{stem}"


def average_hash(image_path: Path, size: int = 8) -> str:
    with Image.open(image_path) as image:
        gray = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(gray.tobytes())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if pixel >= avg else "0" for pixel in pixels)


def hamming(left: str, right: str) -> int:
    return sum(1 for a, b in zip(left, right) if a != b) + abs(len(left) - len(right))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_candidates(dataset_root: Path) -> tuple[list[dict[str, str]], list[DuplicateCandidate]]:
    rows = read_manifest(dataset_root / "clean_manifest.csv")
    candidates: list[DuplicateCandidate] = []
    for index, row in enumerate(rows):
        image_path = Path(row["exported_image"])
        if not image_path.exists():
            image_path = dataset_root / "images" / row["split"] / Path(row["exported_image"]).name
        if not image_path.exists():
            continue
        candidates.append(
            DuplicateCandidate(
                row_index=index,
                image_name=image_path.name,
                class_name=row["class"],
                batch_id=row["batch_id"],
                source_key=normalize_source_key(row),
                hash_value=average_hash(image_path),
                action=row.get("action", ""),
                image_path=str(image_path),
                label_path=row["exported_label"],
            )
        )
    return rows, candidates


def group_duplicate_candidates(
    candidates: list[DuplicateCandidate],
    hash_threshold: int = 4,
) -> list[list[DuplicateCandidate]]:
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i, left in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            right = candidates[j]
            if left.class_name != right.class_name:
                continue
            if left.source_key == right.source_key or hamming(left.hash_value, right.hash_value) <= hash_threshold:
                union(i, j)

    groups_by_root: dict[int, list[DuplicateCandidate]] = {}
    for index, candidate in enumerate(candidates):
        groups_by_root.setdefault(find(index), []).append(candidate)
    return [group for group in groups_by_root.values() if len(group) > 1]


def group_strong_source_duplicates(candidates: list[DuplicateCandidate]) -> list[list[DuplicateCandidate]]:
    grouped: dict[tuple[str, str], list[DuplicateCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.class_name, candidate.source_key), []).append(candidate)
    return [group for group in grouped.values() if len(group) > 1]


def batch_rank(batch_id: str) -> int:
    try:
        return int(batch_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def action_rank(action: str) -> int:
    if action.startswith("manual:"):
        return 30
    if "merge" in action:
        return 20
    if action in {"keep", "kept_original"}:
        return 10
    return 0


def choose_duplicate_keep(group: list[DuplicateCandidate]) -> DuplicateCandidate:
    return max(
        group,
        key=lambda item: (
            item.batch_id == "batch_005",
            batch_rank(item.batch_id),
            action_rank(item.action),
            item.image_name,
        ),
    )


def duplicate_reason(group: list[DuplicateCandidate]) -> str:
    source_keys = {item.source_key for item in group}
    if len(source_keys) == 1:
        return "source_name_match"
    min_distance = min(
        hamming(left.hash_value, right.hash_value)
        for index, left in enumerate(group)
        for right in group[index + 1 :]
    )
    return f"near_image_hash:{min_distance}"


def hash_only_review_groups(
    candidates: list[DuplicateCandidate],
    strong_groups: list[list[DuplicateCandidate]],
    hash_threshold: int,
) -> list[list[DuplicateCandidate]]:
    strong_keys = {frozenset(item.row_index for item in group) for group in strong_groups}
    groups = group_duplicate_candidates(candidates, hash_threshold=hash_threshold)
    review_groups: list[list[DuplicateCandidate]] = []
    seen: set[frozenset[int]] = set()
    for group in groups:
        key = frozenset(item.row_index for item in group)
        if key in strong_keys or key in seen:
            continue
        seen.add(key)
        # Very large hash groups are usually repeated visual backgrounds, not true duplicates.
        if len(group) > 8:
            group = sorted(group, key=lambda item: (item.batch_id, item.image_name))[:8]
        review_groups.append(group)
    return review_groups




def copy_dataset_without_duplicates(
    dataset_root: Path,
    output_root: Path,
    rows: list[dict[str, str]],
    drop_indexes: set[int],
) -> list[dict[str, str]]:
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_root}")
    output_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(rows):
        if row_index in drop_indexes:
            continue
        split = row["split"]
        image_source = Path(row["exported_image"])
        label_source = Path(row["exported_label"])
        if not image_source.exists():
            image_source = dataset_root / "images" / split / Path(row["exported_image"]).name
        if not label_source.exists():
            label_source = dataset_root / "labels" / split / Path(row["exported_label"]).name
        image_target = output_root / "images" / split / image_source.name
        label_target = output_root / "labels" / split / label_source.name
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_source, image_target)
        shutil.copy2(label_source, label_target)
        new_row = dict(row)
        new_row["exported_image"] = str(image_target)
        new_row["exported_label"] = str(label_target)
        output_rows.append(new_row)

    for split in ["train", "val", "test"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    write_data_yaml(output_root)
    write_manifest(output_root / "clean_manifest.csv", output_rows)
    return output_rows


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


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(review_dir: Path, group_index: int, group: list[DuplicateCandidate], keep: DuplicateCandidate) -> str:
    thumbs: list[Image.Image] = []
    tile_w, tile_h = 260, 220
    for item in group:
        with Image.open(item.image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((tile_w, tile_h - 42), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_w, tile_h), "white")
            x = (tile_w - image.width) // 2
            tile.paste(image, (x, 0))
            draw = ImageDraw.Draw(tile)
            status = "KEEP" if item == keep else "DROP"
            color = (20, 130, 60) if item == keep else (190, 40, 40)
            draw.text((8, tile_h - 38), f"{status} {item.batch_id}", fill=color)
            draw.text((8, tile_h - 20), item.image_name[:36], fill=(30, 30, 30))
            thumbs.append(tile)
    sheet = Image.new("RGB", (tile_w * len(thumbs), tile_h), "white")
    for index, tile in enumerate(thumbs):
        sheet.paste(tile, (index * tile_w, 0))
    path = review_dir / "contact_sheets" / f"duplicate_group_{group_index:03d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
    return str(path.relative_to(review_dir)).replace("\\", "/")


def write_review(
    review_dir: Path,
    groups: list[list[DuplicateCandidate]],
    keep_by_group: list[DuplicateCandidate],
    dropped: set[int],
    output_root: Path,
    hash_review_groups: list[list[DuplicateCandidate]],
) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    html_rows: list[str] = []
    for group_index, (group, keep) in enumerate(zip(groups, keep_by_group), 1):
        reason = "source_name_match"
        sheet = make_contact_sheet(review_dir, group_index, group, keep)
        for item in group:
            rows.append(
                {
                    "group_id": f"dup_{group_index:03d}",
                    "decision": "keep" if item == keep else "drop",
                    "reason": reason,
                    "image_name": item.image_name,
                    "class": item.class_name,
                    "batch_id": item.batch_id,
                    "action": item.action,
                    "source_key": item.source_key,
                    "image_path": item.image_path,
                    "contact_sheet": sheet,
                }
            )
        html_rows.append(
            "<tr>"
            f"<td>dup_{group_index:03d}</td>"
            f"<td>{html.escape(reason)}</td>"
            f"<td>{html.escape(keep.image_name)}<br>{html.escape(keep.batch_id)}</td>"
            f"<td>{len(group) - 1}</td>"
            f"<td><img src=\"{html.escape(sheet)}\" loading=\"lazy\"></td>"
            "</tr>"
        )

    with (review_dir / "duplicate_groups.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "group_id",
            "decision",
            "reason",
            "image_name",
            "class",
            "batch_id",
            "action",
            "source_key",
            "image_path",
            "contact_sheet",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    hash_rows: list[dict[str, str]] = []
    hash_html_rows: list[str] = []
    for group_index, group in enumerate(hash_review_groups, 1):
        keep = choose_duplicate_keep(group)
        reason = duplicate_reason(group)
        sheet = make_contact_sheet(review_dir, group_index + 1000, group, keep)
        for item in group:
            hash_rows.append(
                {
                    "group_id": f"similar_{group_index:03d}",
                    "suggestion": "review_keep_candidate" if item == keep else "review_possible_duplicate",
                    "reason": reason,
                    "image_name": item.image_name,
                    "class": item.class_name,
                    "batch_id": item.batch_id,
                    "action": item.action,
                    "source_key": item.source_key,
                    "image_path": item.image_path,
                    "contact_sheet": sheet,
                }
            )
        hash_html_rows.append(
            "<tr>"
            f"<td>similar_{group_index:03d}</td>"
            f"<td>{html.escape(reason)}</td>"
            f"<td>{html.escape(keep.image_name)}<br>{html.escape(keep.batch_id)}</td>"
            f"<td>{len(group)}</td>"
            f"<td><img src=\"{html.escape(sheet)}\" loading=\"lazy\"></td>"
            "</tr>"
        )
    if hash_rows:
        with (review_dir / "similar_hash_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            fieldnames = [
                "group_id",
                "suggestion",
                "reason",
                "image_name",
                "class",
                "batch_id",
                "action",
                "source_key",
                "image_path",
                "contact_sheet",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(hash_rows)

    summary = f"""# v4 重复样本分析

生成时间：2026-06-24

## 结论

- 强重复组：{len(groups)}
- 自动去重样本：{len(dropped)}
- 哈希疑似重复组：{len(hash_review_groups)}
- 去重候选数据集：`{output_root.as_posix()}`

## 规则

1. 同一组重复/近似重复样本中，优先保留 `batch_005`。
2. 如果没有 `batch_005`，优先保留批次号更大的样本。
3. 批次相同则优先保留人工修正或合并后的样本。
4. 只在同一类别内去重，避免不同缺陷类别被误合并。
5. 自动删除只接受同源文件名/来源路径强匹配。
6. 图像哈希只作为疑似重复提示，不自动删除。

## 输出

- `duplicate_groups.csv`：强重复组的保留/删除建议。
- `similar_hash_review.csv`：哈希疑似重复组，仅供人工复核。
- `index.html`：重复组可视化。
- `contact_sheets/`：每组重复图拼图。
"""
    (review_dir / "README.md").write_text(summary, encoding="utf-8")

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>v4 重复样本分析</title>
<style>
body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 24px; color: #172033; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #d8dee9; padding: 8px; vertical-align: top; text-align: left; }}
img {{ max-width: 1100px; width: 100%; height: auto; border: 1px solid #ccd3df; }}
</style>
</head>
<body>
<h1>v4 重复样本分析</h1>
<p>优先保留 batch_005；自动去重只接受同源强匹配；哈希相似只进入复核。</p>
<h2>强重复，已用于 v4 自动去重</h2>
<table>
<thead><tr><th>组</th><th>原因</th><th>保留</th><th>删除数</th><th>对照图</th></tr></thead>
<tbody>{''.join(html_rows)}</tbody>
</table>
<h2>哈希疑似重复，仅供人工复核</h2>
<table>
<thead><tr><th>组</th><th>原因</th><th>建议保留</th><th>组内数量</th><th>对照图</th></tr></thead>
<tbody>{''.join(hash_html_rows)}</tbody>
</table>
</body>
</html>
"""
    (review_dir / "index.html").write_text(document, encoding="utf-8")


def build_duplicate_review(dataset_root: Path, output_root: Path, review_dir: Path, hash_threshold: int) -> None:
    rows, candidates = load_candidates(dataset_root)
    groups = group_strong_source_duplicates(candidates)
    hash_review_groups = hash_only_review_groups(candidates, groups, hash_threshold=hash_threshold)
    keep_by_group = [choose_duplicate_keep(group) for group in groups]
    dropped = {item.row_index for group, keep in zip(groups, keep_by_group) for item in group if item != keep}
    copy_dataset_without_duplicates(dataset_root, output_root, rows, dropped)
    write_review(review_dir, groups, keep_by_group, dropped, output_root, hash_review_groups)
    print(f"groups: {len(groups)}")
    print(f"dropped: {len(dropped)}")
    print(f"hash_review_groups: {len(hash_review_groups)}")
    print(review_dir / "index.html")
    print(output_root)


def main() -> None:
    args = parse_args()
    build_duplicate_review(args.dataset_root, args.output_root, args.review_dir, args.hash_threshold)


if __name__ == "__main__":
    main()
