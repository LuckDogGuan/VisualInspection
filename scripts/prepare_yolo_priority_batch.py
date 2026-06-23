from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


CLASS_ORDER = ["dent", "powder", "stain", "crack", "transverse_bump"]

CLASS_SOURCES = {
    "dent": {
        "true_class_cn": "碰伤",
        "source_dir": Path("data/ali2018/Be injured by a collision"),
    },
    "powder": {
        "true_class_cn": "凸粉",
        "source_dir": Path("data/ali2018/Convex powder"),
    },
    "stain": {
        "true_class_cn": "脏点",
        "source_dir": Path("data/ali2018/Dirty spot"),
    },
    "crack": {
        "true_class_cn": "涂层开裂",
        "source_dir": Path("data/ali2018/Coating cracking"),
    },
    "transverse_bump": {
        "true_class_cn": "横条压凹",
        "source_dir": Path("data/ali2018/The transverse strip is dented"),
    },
}

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class SourceImage:
    class_name: str
    true_class_cn: str
    source_path: Path
    confidence: str
    pred_class_cn: str
    correct: str


def read_registry(registry_path: Path) -> list[dict[str, str]]:
    if not registry_path.exists():
        return []
    with registry_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_registry(registry_path: Path, rows: list[dict[str, str]]) -> None:
    with registry_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def collect_used_source_names(stage_root: Path, registry_rows: list[dict[str, str]]) -> set[str]:
    used: set[str] = set()
    for row in registry_rows:
        source_path = row.get("source_path", "")
        if source_path:
            used.add(Path(source_path).name)

    image_dirs = [stage_root / "labelimg_work" / "images"]
    batches_dir = stage_root / "batches"
    if batches_dir.exists():
        image_dirs.extend(p / "images" for p in batches_dir.glob("batch_*") if p.is_dir())

    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        for image_path in image_dir.iterdir():
            if not image_path.is_file():
                continue
            source_name = image_path.name.split("__", 1)[-1]
            used.add(source_name)
    return used


def load_confidence_lookup(report_path: Path) -> dict[str, dict[str, str]]:
    if not report_path.exists():
        return {}

    lookup: dict[str, dict[str, str]] = {}
    with report_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            source_path = Path(row.get("source_path", ""))
            if not source_path.name:
                continue
            lookup[source_path.name] = {
                "confidence": row.get("confidence", ""),
                "pred_class_cn": row.get("pred_class_cn", ""),
                "correct": row.get("correct", ""),
            }
    return lookup


def parse_counts(raw_counts: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in raw_counts:
        if "=" not in item:
            raise ValueError(f"Invalid count item: {item}. Expected class=count.")
        class_name, count_text = item.split("=", 1)
        class_name = class_name.strip()
        if class_name not in CLASS_SOURCES:
            raise ValueError(f"Unsupported class: {class_name}")
        count = int(count_text)
        if count <= 0:
            raise ValueError(f"Count must be positive: {item}")
        counts[class_name] = count
    return counts


def list_source_images(
    project_root: Path,
    counts: dict[str, int],
    used_source_names: set[str],
    confidence_lookup: dict[str, dict[str, str]],
    seed: int,
) -> list[SourceImage]:
    rng = random.Random(seed)
    selected: list[SourceImage] = []

    for class_name in CLASS_ORDER:
        count = counts.get(class_name, 0)
        if count <= 0:
            continue

        source_info = CLASS_SOURCES[class_name]
        source_dir = project_root / source_info["source_dir"]
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)

        candidates = [
            p
            for p in source_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and p.name not in used_source_names
        ]
        candidates.sort(key=lambda p: p.name)
        if len(candidates) < count:
            raise RuntimeError(
                f"Not enough unused images for {class_name}: need {count}, got {len(candidates)}"
            )

        # Take a deterministic random sample so batches are not clustered by filename time.
        for source_path in rng.sample(candidates, count):
            confidence_info = confidence_lookup.get(source_path.name, {})
            selected.append(
                SourceImage(
                    class_name=class_name,
                    true_class_cn=str(source_info["true_class_cn"]),
                    source_path=source_path,
                    confidence=confidence_info.get("confidence", ""),
                    pred_class_cn=confidence_info.get("pred_class_cn", ""),
                    correct=confidence_info.get("correct", ""),
                )
            )
    return selected


def write_class_files(batch_root: Path) -> None:
    class_text = "\n".join(CLASS_ORDER) + "\n"
    (batch_root / "classes.txt").write_text(class_text, encoding="utf-8")
    (batch_root / "predefined_classes.txt").write_text(class_text, encoding="utf-8")
    labels_dir = batch_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "classes.txt").write_text(class_text, encoding="utf-8")


def copy_batch(
    stage_root: Path,
    batch_id: str,
    selected: list[SourceImage],
    notes: str,
) -> list[dict[str, str]]:
    batch_root = stage_root / "batches" / batch_id
    images_dir = batch_root / "images"
    labels_dir = batch_root / "labels"

    if batch_root.exists():
        existing_files = [p for p in batch_root.rglob("*") if p.is_file()]
        if existing_files:
            raise RuntimeError(f"Batch directory already contains files: {batch_root}")

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    write_class_files(batch_root)

    now = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, str]] = []
    for image in selected:
        batch_image_name = f"{image.class_name}__{image.source_path.name}"
        batch_image_path = images_dir / batch_image_name
        batch_label_path = labels_dir / f"{batch_image_path.stem}.txt"
        shutil.copy2(image.source_path, batch_image_path)

        row_notes = notes
        if image.pred_class_cn or image.correct:
            row_notes = (
                f"{notes}; classifier_pred={image.pred_class_cn}; "
                f"classifier_correct={image.correct}"
            )

        rows.append(
            {
                "batch_id": batch_id,
                "status": "selected",
                "class": image.class_name,
                "true_class_cn": image.true_class_cn,
                "confidence": image.confidence,
                "source_path": str(image.source_path),
                "batch_image_path": str(batch_image_path),
                "batch_label_path": str(batch_label_path),
                "selected_at": now,
                "annotated_at": "",
                "notes": row_notes,
            }
        )
    return rows


def write_manifest(batch_root: Path, rows: list[dict[str, str]]) -> None:
    manifest_path = batch_root / "selection_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--count", action="append", required=True, help="Example: stain=60")
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument(
        "--confidence-report",
        default="outputs/classification_results/full_screening/labeled_all_report.csv",
    )
    parser.add_argument(
        "--notes",
        default="selected from standard class folders; source class trusted over weak full-screen classifier",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage_root = project_root / "data" / "yolo_stage3_manual"
    registry_path = stage_root / "annotation_registry.csv"
    batch_root = stage_root / "batches" / args.batch_id

    registry_rows = read_registry(registry_path)
    used_source_names = collect_used_source_names(stage_root, registry_rows)
    confidence_lookup = load_confidence_lookup(project_root / args.confidence_report)
    counts = parse_counts(args.count)
    selected = list_source_images(project_root, counts, used_source_names, confidence_lookup, args.seed)
    batch_rows = copy_batch(stage_root, args.batch_id, selected, args.notes)

    registry_rows.extend(batch_rows)
    write_registry(registry_path, registry_rows)
    write_manifest(batch_root, batch_rows)

    print(f"batch_id={args.batch_id}")
    print(f"images={len(batch_rows)}")
    for class_name in CLASS_ORDER:
        class_rows = [r for r in batch_rows if r["class"] == class_name]
        if class_rows:
            known_conf = [r["confidence"] for r in class_rows if r["confidence"]]
            print(f"{class_name}: {len(class_rows)} confidence_recorded={len(known_conf)}")
    print(batch_root)


if __name__ == "__main__":
    main()
