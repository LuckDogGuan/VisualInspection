from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_yolo_annotation_review_guides import (
    CLASS_ORDER,
    apply_manual_decision,
    read_yolo_boxes,
    suggest_boxes,
)


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


def ensure_empty_output(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {output_root}\n"
            "Use a new --output-name so the clean candidate is reproducible."
        )


def split_rows(rows: list[dict[str, str]], val_ratio: float, seed: int) -> dict[str, list[dict[str, str]]]:
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)

    rng = random.Random(seed)
    splits = {"train": [], "val": []}
    for class_name in CLASS_ORDER:
        class_rows = list(by_class[class_name])
        rng.shuffle(class_rows)
        if val_ratio <= 0 or len(class_rows) <= 1:
            val_count = 0
        else:
            val_count = max(1, round(len(class_rows) * val_ratio))
        splits["val"].extend(class_rows[:val_count])
        splits["train"].extend(class_rows[val_count:])
    return splits


def write_yolo_label(path: Path, boxes) -> None:
    lines = [
        f"{box.class_id} {box.cx:.6f} {box.cy:.6f} {box.width:.6f} {box.height:.6f}"
        for box in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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


def candidate_label_for_row(
    row: dict[str, str],
    problem_batches: set[str],
    problem_classes: set[str],
    decisions: dict[str, dict[str, str]] | None = None,
):
    boxes = read_yolo_boxes(Path(row["batch_label_path"]))
    decision_row = (decisions or {}).get(path_key(row["batch_label_path"]))
    if decision_row is not None:
        suggested, _, _ = suggest_boxes(row["class"], boxes)
        adjusted, action, reason = apply_manual_decision(boxes, suggested, decision_row["decision"])
        notes = decision_row.get("notes", "")
        if notes:
            reason = f"{reason} {notes}"
        return adjusted, action, reason

    if row["batch_id"] in problem_batches and row["class"] in problem_classes:
        suggested, action, reason = suggest_boxes(row["class"], boxes)
        if not suggested:
            return [], action, reason
        note = "merge_candidate" if action == "merge" else action
        return suggested, note, reason
    return boxes, "kept_original", "原样保留，未纳入本轮问题批次自动候选调整。"


def build_clean_dataset(
    stage_root: Path,
    output_name: str,
    problem_batches: set[str],
    problem_classes: set[str],
    val_ratio: float,
    seed: int,
    decisions_path: Path | None = None,
) -> Path:
    registry_path = stage_root / "annotation_registry.csv"
    output_root = stage_root / "exports" / output_name
    ensure_empty_output(output_root)

    rows = [
        row
        for row in read_registry(registry_path)
        if row.get("status") == "annotated"
        and row.get("class") in CLASS_ORDER
        and Path(row.get("batch_image_path", "")).exists()
        and Path(row.get("batch_label_path", "")).exists()
    ]
    if not rows:
        raise SystemExit("No annotated rows found.")

    decisions = read_decisions(decisions_path)
    splits = split_rows(rows, val_ratio=val_ratio, seed=seed)
    manifest_rows: list[dict[str, str]] = []
    for split in ["train", "val"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(splits[split], 1):
            source_image = Path(row["batch_image_path"])
            exported_image_name = f"{row['class']}_{row['batch_id']}_{index:04d}{source_image.suffix.lower()}"
            exported_label_name = f"{Path(exported_image_name).stem}.txt"
            exported_image_path = output_root / "images" / split / exported_image_name
            exported_label_path = output_root / "labels" / split / exported_label_name

            candidate_boxes, action, reason = candidate_label_for_row(row, problem_batches, problem_classes, decisions)
            if not candidate_boxes:
                continue
            shutil.copy2(source_image, exported_image_path)
            if action == "kept_original":
                shutil.copy2(Path(row["batch_label_path"]), exported_label_path)
            else:
                write_yolo_label(exported_label_path, candidate_boxes)
            manifest_rows.append(
                {
                    "split": split,
                    "class": row["class"],
                    "true_class_cn": row.get("true_class_cn", ""),
                    "batch_id": row["batch_id"],
                    "action": action,
                    "reason": reason,
                    "box_count_clean": str(len(candidate_boxes)),
                    "exported_image": str(exported_image_path),
                    "exported_label": str(exported_label_path),
                    "source_path": row.get("source_path", ""),
                    "batch_image_path": row["batch_image_path"],
                    "batch_label_path": row["batch_label_path"],
                    "original_filename": source_image.name,
                }
            )

    (output_root / "images" / "test").mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / "test").mkdir(parents=True, exist_ok=True)
    write_data_yaml(output_root)
    if not manifest_rows:
        raise SystemExit("No clean candidate rows were exported.")

    with (output_root / "clean_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a non-destructive clean candidate YOLO dataset.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-name", default="dataset_clean_candidate_v1")
    parser.add_argument("--problem-batches", nargs="+", default=["batch_004", "batch_005"])
    parser.add_argument("--problem-classes", nargs="+", default=["stain", "dent", "powder"])
    parser.add_argument(
        "--decisions",
        type=Path,
        default=None,
        help="Optional manual_review_decisions.csv. Defaults to data/yolo_stage3_manual/review/manual_review_decisions.csv when present.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260623)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_root = args.project_root.resolve() / "data" / "yolo_stage3_manual"
    decisions_path = args.decisions or stage_root / "review" / "manual_review_decisions.csv"
    output_root = build_clean_dataset(
        stage_root=stage_root,
        output_name=args.output_name,
        problem_batches=set(args.problem_batches),
        problem_classes=set(args.problem_classes),
        val_ratio=args.val_ratio,
        seed=args.seed,
        decisions_path=decisions_path,
    )
    manifest_path = output_root / "clean_manifest.csv"
    print(f"output_root={output_root}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
