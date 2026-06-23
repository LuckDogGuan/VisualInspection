from __future__ import annotations

import argparse
import csv
from datetime import datetime
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


def read_registry(registry_path: Path) -> list[dict[str, str]]:
    with registry_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_registry(registry_path: Path, rows: list[dict[str, str]]) -> None:
    with registry_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_label(label_path: Path, allow_missing: bool) -> tuple[int, list[str], str]:
    errors: list[str] = []
    if not label_path.exists():
        if allow_missing:
            return 0, [], "missing label; skipped because no confirmed defect was annotated"
        return 0, [f"missing label: {label_path}"], ""

    text = label_path.read_text(encoding="utf-8-sig")
    # Normalize labels to UTF-8 without BOM for Ultralytics.
    label_path.write_text(text, encoding="utf-8")

    box_count = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path.name}:{line_no}: expected 5 columns, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError as exc:
            errors.append(f"{label_path.name}:{line_no}: {exc}")
            continue
        if class_id < 0 or class_id >= len(CLASS_ORDER):
            errors.append(f"{label_path.name}:{line_no}: class id out of range: {class_id}")
        if any(value < 0 or value > 1 for value in coords):
            errors.append(f"{label_path.name}:{line_no}: coordinate out of 0..1: {coords}")
        box_count += 1
    if box_count == 0:
        if allow_missing:
            return 0, [], "empty label; skipped because no confirmed defect was annotated"
        errors.append(f"{label_path.name}: empty label")
    return box_count, errors, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=r"D:\code\VisualInspection\铝型材缺陷图")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Skip missing or empty labels instead of failing the batch.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    stage_root = project_root / "data" / "yolo_stage3_manual"
    registry_path = stage_root / "annotation_registry.csv"
    rows = read_registry(registry_path)

    batch_rows = [row for row in rows if row.get("batch_id") == args.batch_id]
    if not batch_rows:
        raise SystemExit(f"No registry rows for {args.batch_id}")

    errors: list[str] = []
    total_boxes = 0
    annotated_count = 0
    skipped_count = 0
    row_updates: dict[str, tuple[str, int, str]] = {}
    for row in batch_rows:
        box_count, label_errors, skip_note = validate_label(
            Path(row["batch_label_path"]),
            allow_missing=args.allow_missing_labels,
        )
        total_boxes += box_count
        errors.extend(label_errors)
        if box_count > 0:
            annotated_count += 1
            row_updates[row["batch_label_path"]] = ("annotated", box_count, "")
        else:
            skipped_count += 1
            row_updates[row["batch_label_path"]] = ("skipped", 0, skip_note)

    if errors:
        print("Validation failed:")
        for error in errors[:50]:
            print(error)
        raise SystemExit(1)

    now = datetime.now().isoformat(timespec="seconds")
    for row in rows:
        if row.get("batch_id") == args.batch_id:
            status, box_count, skip_note = row_updates[row["batch_label_path"]]
            row["status"] = status
            row["annotated_at"] = now if status == "annotated" else ""
            if status == "skipped":
                existing_notes = row.get("notes", "")
                row["notes"] = f"{existing_notes}; {skip_note}".strip("; ")

    write_registry(registry_path, rows)
    print(f"batch_id={args.batch_id}")
    print(f"images={len(batch_rows)}")
    print(f"annotated_images={annotated_count}")
    print(f"skipped_images={skipped_count}")
    print(f"boxes={total_boxes}")
    print("status=finalized")


if __name__ == "__main__":
    main()
