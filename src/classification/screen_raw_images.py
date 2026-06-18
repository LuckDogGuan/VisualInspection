from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.config import CLASS_ID_TO_CN, get_config, resolve_path
from classification.data import is_image
from classification.inference import load_model_for_inference, resolve_inference_image_size
from classification.train import build_eval_transforms


@dataclass(frozen=True)
class RawImageRow:
    image_path: Path
    source_name: str


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def collect_raw_images(input_root: Path, source_names: Sequence[str] = ("APSPC1", "APSPC2")) -> list[RawImageRow]:
    rows: list[RawImageRow] = []
    for source_name in source_names:
        source_dir = input_root / source_name
        if not source_dir.exists():
            continue
        for image_path in sorted(source_dir.rglob("*"), key=lambda item: item.as_posix().lower()):
            if is_image(image_path):
                rows.append(RawImageRow(image_path=image_path, source_name=source_name))
    return rows


def should_copy_as_nice_picture(prediction: dict, normal_class_id: int = 0, min_confidence: float = 0.0) -> bool:
    if prediction.get("status", "ok") != "ok":
        return False
    return int(prediction["class_id"]) == normal_class_id and float(prediction["confidence"]) >= min_confidence


def class_name_for_file(class_id: int, class_name: str) -> str:
    if class_id == 0:
        return "normal"
    return safe_name(class_name)


def build_nice_picture_target(
    output_root: Path,
    source_name: str,
    image_path: Path,
    class_id: int,
    class_name: str,
    confidence: float,
) -> Path:
    pred_name = f"{class_id:02d}_{class_name_for_file(class_id, class_name)}"
    output_name = f"{image_path.stem}__pred_{pred_name}__conf_{confidence:.3f}{image_path.suffix}"
    return output_root / source_name / output_name


def _error_record(row: RawImageRow, exc: Exception) -> dict:
    return {
        "source_name": row.source_name,
        "source_path": str(row.image_path),
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "class_id": "",
        "class_name_cn": "",
        "confidence": "",
        "top3_class_ids": "",
        "top3_class_cn": "",
        "top3_confidence": "",
        "copied_path": "",
    }


def predict_raw_images(model, rows: Sequence[RawImageRow], image_size: int, device: torch.device) -> list[dict]:
    transform = build_eval_transforms(image_size)
    records: list[dict] = []
    for index, row in enumerate(rows, start=1):
        try:
            image = Image.open(row.image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()

            top_values, top_ids = torch.topk(probabilities, k=min(3, probabilities.numel()))
            pred_id = int(top_ids[0].item())
            pred_cn = CLASS_ID_TO_CN[pred_id]
            confidence = float(top_values[0].item())
            records.append(
                {
                    "source_name": row.source_name,
                    "source_path": str(row.image_path),
                    "status": "ok",
                    "error": "",
                    "class_id": pred_id,
                    "class_name_cn": pred_cn,
                    "confidence": confidence,
                    "top3_class_ids": "|".join(str(int(item.item())) for item in top_ids),
                    "top3_class_cn": "|".join(CLASS_ID_TO_CN[int(item.item())] for item in top_ids),
                    "top3_confidence": "|".join(f"{float(item.item()):.6f}" for item in top_values),
                    "copied_path": "",
                }
            )
        except Exception as exc:
            records.append(_error_record(row, exc))

        if index % 200 == 0 or index == len(rows):
            print(f"processed={index}/{len(rows)}")
    return records


def copy_nice_pictures(
    records: Sequence[dict],
    output_root: Path,
    normal_class_id: int,
    min_confidence: float,
) -> int:
    copied = 0
    for record in records:
        if not should_copy_as_nice_picture(record, normal_class_id=normal_class_id, min_confidence=min_confidence):
            continue
        source_path = Path(str(record["source_path"]))
        target = build_nice_picture_target(
            output_root=output_root,
            source_name=str(record["source_name"]),
            image_path=source_path,
            class_id=int(record["class_id"]),
            class_name=str(record["class_name_cn"]),
            confidence=float(record["confidence"]),
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        record["copied_path"] = str(target)
        copied += 1
    return copied


def write_predictions_csv(records: Sequence[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_name",
        "source_path",
        "status",
        "error",
        "class_id",
        "class_name_cn",
        "confidence",
        "top3_class_ids",
        "top3_class_cn",
        "top3_confidence",
        "copied_path",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary(records: Sequence[dict], output_md: Path, copied_count: int, normal_class_id: int, min_confidence: float) -> None:
    by_source: dict[str, int] = {}
    normal_by_source: dict[str, int] = {}
    error_by_source: dict[str, int] = {}
    by_class: dict[int, int] = {}
    for record in records:
        source_name = str(record["source_name"])
        by_source[source_name] = by_source.get(source_name, 0) + 1
        if record.get("status", "ok") != "ok":
            error_by_source[source_name] = error_by_source.get(source_name, 0) + 1
            continue
        class_id = int(record["class_id"])
        by_class[class_id] = by_class.get(class_id, 0) + 1
        if should_copy_as_nice_picture(record, normal_class_id=normal_class_id, min_confidence=min_confidence):
            normal_by_source[source_name] = normal_by_source.get(source_name, 0) + 1

    error_count = sum(1 for record in records if record.get("status", "ok") != "ok")
    lines = [
        "# raw_images screening summary",
        "",
        "## Machine-readable summary",
        "",
        f"total_images={len(records)}",
        f"copied_to_nice_picture={copied_count}",
        f"error_images={error_count}",
        f"normal_class_id={normal_class_id}",
        f"min_confidence={min_confidence:.2f}",
        "",
        "## By source",
        "",
        "| source | images | copied_predicted_normal | read_errors |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source_name in sorted(by_source):
        lines.append(
            f"| {source_name} | {by_source[source_name]} | "
            f"{normal_by_source.get(source_name, 0)} | {error_by_source.get(source_name, 0)} |"
        )

    lines.extend(["", "## By predicted class", "", "| class_id | class_name | images |", "| ---: | --- | ---: |"])
    for class_id in sorted(by_class):
        lines.append(f"| {class_id} | {CLASS_ID_TO_CN[class_id]} | {by_class[class_id]} |")

    lines.extend(
        [
            "",
            "## Manual review notes",
            "",
            "- Files under nice_picture are images predicted as normal by the model.",
            "- If manual review finds real defects in nice_picture, move those files into a missed-defect set for recall-focused retraining.",
            "- This script copies images only. It never moves or deletes source images.",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen supplier raw images and copy predicted-normal images for manual review")
    parser.add_argument("--env", choices=["auto", "windows", "linux"], default="auto")
    parser.add_argument("--model", type=Path, default=Path("outputs/classification_results/deploy/classifier.torchscript.pt"))
    parser.add_argument("--input-root", type=Path, default=Path("data/raw_images"))
    parser.add_argument("--output-root", type=Path, default=Path("data/raw_images/nice_picture"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/classification_results/raw_images_screening"))
    parser.add_argument("--source", action="append", dest="sources", default=None, help="Source folder name, repeatable")
    parser.add_argument("--limit", type=int, default=None, help="Limit images for local smoke tests")
    parser.add_argument("--normal-class-id", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--no-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.env)
    model_path = resolve_path(config, args.model)
    input_root = resolve_path(config, args.input_root)
    output_root = resolve_path(config, args.output_root)
    report_dir = resolve_path(config, args.report_dir)
    sources = tuple(args.sources or ("APSPC1", "APSPC2"))

    rows = collect_raw_images(input_root, source_names=sources)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    if not rows:
        raise RuntimeError(f"No raw images found under {input_root} for sources: {', '.join(sources)}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(model_path, device)
    image_size = resolve_inference_image_size(model_path, config.image_size)
    records = predict_raw_images(model, rows, image_size=image_size, device=device)

    copied_count = 0
    if not args.no_copy:
        copied_count = copy_nice_pictures(records, output_root, normal_class_id=args.normal_class_id, min_confidence=args.min_confidence)

    report_csv = report_dir / "raw_predictions.csv"
    summary_md = report_dir / "summary.md"
    write_predictions_csv(records, report_csv)
    write_summary(records, summary_md, copied_count, normal_class_id=args.normal_class_id, min_confidence=args.min_confidence)

    error_count = sum(1 for record in records if record.get("status", "ok") != "ok")
    print(f"images={len(records)}")
    print(f"copied={copied_count}")
    print(f"errors={error_count}")
    print(f"output_root={output_root}")
    print(f"csv={report_csv}")
    print(f"summary={summary_md}")


if __name__ == "__main__":
    main()
