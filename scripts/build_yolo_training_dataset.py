from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


CLASS_ORDER = ["dent", "powder", "stain", "crack", "transverse_bump"]
REGISTRY_FIELDS = [
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
]


def read_registry(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def validate_label(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    if not path.exists():
        return 0, [f"missing label: {path}"]
    text = path.read_text(encoding="utf-8-sig")
    path.write_text(text, encoding="utf-8")
    boxes = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{line_no}: expected 5 columns")
            continue
        try:
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError as exc:
            errors.append(f"{path.name}:{line_no}: {exc}")
            continue
        if class_id < 0 or class_id >= len(CLASS_ORDER):
            errors.append(f"{path.name}:{line_no}: bad class id {class_id}")
        if any(value < 0 or value > 1 for value in coords):
            errors.append(f"{path.name}:{line_no}: coordinate out of 0..1")
        boxes += 1
    if boxes == 0:
        errors.append(f"{path.name}: empty label")
    return boxes, errors


def split_rows(rows: list[dict[str, str]], val_ratio: float, seed: int) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)

    splits = {"train": [], "val": []}
    for class_name in CLASS_ORDER:
        class_rows = by_class[class_name]
        rng.shuffle(class_rows)
        if len(class_rows) <= 1:
            val_count = 0
        else:
            val_count = max(1, round(len(class_rows) * val_ratio))
        splits["val"].extend(class_rows[:val_count])
        splits["train"].extend(class_rows[val_count:])
    return splits


def write_data_yaml(export_root: Path) -> None:
    names = "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_ORDER))
    text = (
        f"path: {export_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names}"
    )
    (export_root / "data.yaml").write_text(text, encoding="utf-8")


def ensure_clean_export_root(export_root: Path) -> None:
    if export_root.exists() and any(export_root.iterdir()):
        raise SystemExit(
            f"Export directory is not empty: {export_root}\n"
            "Use a new --output-name to create a clean dataset."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--output-name", default="dataset_annotated")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260622)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    stage_root = project_root / "data" / "yolo_stage3_manual"
    registry_path = stage_root / "annotation_registry.csv"
    export_root = stage_root / "exports" / args.output_name
    ensure_clean_export_root(export_root)

    rows = [
        row
        for row in read_registry(registry_path)
        if row.get("status") == "annotated" and row.get("class") in CLASS_ORDER
    ]
    if not rows:
        raise SystemExit("No annotated rows in annotation_registry.csv")

    errors: list[str] = []
    for row in rows:
        image_path = Path(row["batch_image_path"])
        label_path = Path(row["batch_label_path"])
        if not image_path.exists():
            errors.append(f"missing image: {image_path}")
        _, label_errors = validate_label(label_path)
        errors.extend(label_errors)
    if errors:
        print("Validation failed:")
        for error in errors[:100]:
            print(error)
        raise SystemExit(1)

    splits = split_rows(rows, args.val_ratio, args.seed)
    manifest_rows: list[dict[str, str]] = []
    for split, split_rows_value in splits.items():
        (export_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (export_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(split_rows_value, 1):
            image_path = Path(row["batch_image_path"])
            label_path = Path(row["batch_label_path"])
            class_name = row["class"]
            suffix = image_path.suffix.lower()
            exported_name = f"{class_name}_{row['batch_id']}_{index:04d}{suffix}"
            exported_label = f"{Path(exported_name).stem}.txt"
            exported_image_path = export_root / "images" / split / exported_name
            exported_label_path = export_root / "labels" / split / exported_label
            shutil.copy2(image_path, exported_image_path)
            shutil.copy2(label_path, exported_label_path)
            manifest_rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "batch_id": row["batch_id"],
                    "exported_image": str(exported_image_path),
                    "exported_label": str(exported_label_path),
                    "source_path": row["source_path"],
                    "batch_image_path": row["batch_image_path"],
                    "batch_label_path": row["batch_label_path"],
                }
            )

    (export_root / "images" / "test").mkdir(parents=True, exist_ok=True)
    (export_root / "labels" / "test").mkdir(parents=True, exist_ok=True)
    write_data_yaml(export_root)
    with (export_root / "export_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "split",
            "class",
            "batch_id",
            "exported_image",
            "exported_label",
            "source_path",
            "batch_image_path",
            "batch_label_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"export_root={export_root}")
    print(f"images={len(manifest_rows)}")
    for split in ["train", "val"]:
        print(f"{split}={len(splits[split])}")
    for class_name in CLASS_ORDER:
        count = sum(1 for row in manifest_rows if row["class"] == class_name)
        print(f"{class_name}={count}")


if __name__ == "__main__":
    main()
