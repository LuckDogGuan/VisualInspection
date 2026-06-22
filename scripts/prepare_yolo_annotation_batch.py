from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


CLASS_MAP = {
    "碰伤": "dent",
    "凸粉": "powder",
    "脏点": "stain",
    "涂层开裂": "crack",
    "横条压凹": "transverse_bump",
}

CLASS_ORDER = ["dent", "powder", "stain", "crack", "transverse_bump"]
REMOTE_PREFIX = "/home/guanjianxiong/code/yolo/VisualInspection"

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


@dataclass(frozen=True)
class Candidate:
    source_path: Path
    true_class_cn: str
    class_name: str
    confidence: float
    pred_class_cn: str
    correct: str


def normalize_source_path(raw_path: str, project_root: Path) -> Path:
    path = raw_path.strip().replace("\\", "/")
    if path.startswith(REMOTE_PREFIX):
        rel = path.removeprefix(REMOTE_PREFIX).lstrip("/")
        return project_root / rel
    return Path(raw_path)


def read_registry(registry_path: Path) -> list[dict[str, str]]:
    if not registry_path.exists():
        return []
    with registry_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_registry(registry_path: Path, rows: list[dict[str, str]]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_batch_001(stage_root: Path, registry_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if registry_rows:
        return registry_rows

    manifest_path = stage_root / "selection_manifest.csv"
    labels_dir = stage_root / "labelimg_work" / "labels"
    if not manifest_path.exists():
        return registry_rows

    now = datetime.now().isoformat(timespec="seconds")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            image_path = Path(row["labelimg_image"])
            label_path = labels_dir / f"{image_path.stem}.txt"
            registry_rows.append(
                {
                    "batch_id": "batch_001",
                    "status": "annotated" if label_path.exists() else "selected",
                    "class": row["class"],
                    "true_class_cn": row["true_class_cn"],
                    "confidence": row["confidence"],
                    "source_path": str(Path(row["source_path"])),
                    "batch_image_path": str(image_path),
                    "batch_label_path": str(label_path),
                    "selected_at": now,
                    "annotated_at": now if label_path.exists() else "",
                    "notes": "bootstrapped from selection_manifest.csv",
                }
            )
    return registry_rows


def next_batch_id(registry_rows: list[dict[str, str]]) -> str:
    max_id = 1
    for row in registry_rows:
        batch_id = row.get("batch_id", "")
        if batch_id.startswith("batch_"):
            try:
                max_id = max(max_id, int(batch_id.split("_", 1)[1]))
            except ValueError:
                continue
    return f"batch_{max_id + 1:03d}"


def load_candidates(report_path: Path, project_root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    with report_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            true_class_cn = row.get("true_class_cn", "").strip()
            if true_class_cn not in CLASS_MAP:
                continue
            raw_source = row.get("source_path", "")
            if not raw_source:
                continue
            source_path = normalize_source_path(raw_source, project_root)
            if not source_path.exists():
                continue
            try:
                confidence = float(row.get("confidence", "0") or 0)
            except ValueError:
                continue
            candidates.append(
                Candidate(
                    source_path=source_path,
                    true_class_cn=true_class_cn,
                    class_name=CLASS_MAP[true_class_cn],
                    confidence=confidence,
                    pred_class_cn=row.get("pred_class_cn", "").strip(),
                    correct=row.get("correct", "").strip(),
                )
            )
    return candidates


def choose_candidates(
    candidates: list[Candidate],
    used_sources: set[str],
    per_class: int,
    top_pool: int,
    seed: int,
) -> list[Candidate]:
    rng = random.Random(seed)
    selected: list[Candidate] = []
    for class_name in CLASS_ORDER:
        class_candidates = [
            c
            for c in candidates
            if c.class_name == class_name
            and str(c.source_path) not in used_sources
            and c.pred_class_cn == c.true_class_cn
            and c.correct.lower() == "true"
        ]
        class_candidates.sort(key=lambda c: c.confidence, reverse=True)
        pool = class_candidates[: max(per_class, top_pool)]
        if len(pool) < per_class:
            fallback = [
                c
                for c in candidates
                if c.class_name == class_name and str(c.source_path) not in used_sources
            ]
            fallback.sort(key=lambda c: c.confidence, reverse=True)
            pool = fallback[: max(per_class, top_pool)]
        if len(pool) < per_class:
            raise RuntimeError(f"Not enough unused candidates for {class_name}: {len(pool)}")
        selected.extend(rng.sample(pool, per_class))
    return selected


def copy_batch_files(stage_root: Path, batch_id: str, selected: list[Candidate]) -> list[dict[str, str]]:
    batch_root = stage_root / "batches" / batch_id
    images_dir = batch_root / "images"
    labels_dir = batch_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_text = "\n".join(CLASS_ORDER) + "\n"
    (batch_root / "classes.txt").write_text(class_text, encoding="utf-8")
    (labels_dir / "classes.txt").write_text(class_text, encoding="utf-8")
    (batch_root / "predefined_classes.txt").write_text(class_text, encoding="utf-8")

    rows: list[dict[str, str]] = []
    now = datetime.now().isoformat(timespec="seconds")
    for candidate in selected:
        image_name = f"{candidate.class_name}__{candidate.source_path.name}"
        image_path = images_dir / image_name
        label_path = labels_dir / f"{Path(image_name).stem}.txt"
        shutil.copy2(candidate.source_path, image_path)
        rows.append(
            {
                "batch_id": batch_id,
                "status": "selected",
                "class": candidate.class_name,
                "true_class_cn": candidate.true_class_cn,
                "confidence": f"{candidate.confidence:.6f}",
                "source_path": str(candidate.source_path),
                "batch_image_path": str(image_path),
                "batch_label_path": str(label_path),
                "selected_at": now,
                "annotated_at": "",
                "notes": "",
            }
        )
    return rows


def write_batch_manifest(stage_root: Path, batch_id: str, rows: list[dict[str, str]]) -> None:
    manifest_path = stage_root / "batches" / batch_id / "selection_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=r"D:\code\VisualInspection\铝型材缺陷图")
    parser.add_argument(
        "--report",
        default=r"D:\code\VisualInspection\铝型材缺陷图\server_outputs\classification_results\evaluation_samples\labeled_report.csv",
    )
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--top-pool", type=int, default=30)
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--seed", type=int, default=20260622)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    stage_root = project_root / "data" / "yolo_stage3_manual"
    registry_path = stage_root / "annotation_registry.csv"

    registry_rows = bootstrap_batch_001(stage_root, read_registry(registry_path))
    batch_id = args.batch_id or next_batch_id(registry_rows)
    used_sources = {row["source_path"] for row in registry_rows if row.get("source_path")}

    candidates = load_candidates(Path(args.report), project_root)
    selected = choose_candidates(candidates, used_sources, args.per_class, args.top_pool, args.seed)
    batch_rows = copy_batch_files(stage_root, batch_id, selected)

    registry_rows.extend(batch_rows)
    write_registry(registry_path, registry_rows)
    write_batch_manifest(stage_root, batch_id, batch_rows)

    print(f"batch_id={batch_id}")
    print(f"images={len(batch_rows)}")
    for class_name in CLASS_ORDER:
        class_rows = [r for r in batch_rows if r["class"] == class_name]
        confs = [float(r["confidence"]) for r in class_rows]
        print(
            f"{class_name}: {len(class_rows)} "
            f"min_conf={min(confs):.3f} max_conf={max(confs):.3f}"
        )
    print(stage_root / "batches" / batch_id)


if __name__ == "__main__":
    main()
