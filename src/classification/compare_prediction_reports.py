from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping


def prediction_key(row: Mapping[str, str]) -> str:
    source_path = Path(row.get("source_path", ""))
    file_name = source_path.name or row.get("image_path", "")
    source_name = row.get("source_name", "")
    if source_name:
        return f"{source_name}/{file_name}"
    return file_name


def load_prediction_records(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        records: dict[str, dict[str, str]] = {}
        for row in reader:
            records[prediction_key(row)] = dict(row)
    return records


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def compare_records(
    baseline: Mapping[str, Mapping[str, str]],
    current: Mapping[str, Mapping[str, str]],
    confidence_tolerance: float = 0.001,
) -> dict:
    baseline_keys = set(baseline)
    current_keys = set(current)
    common_keys = sorted(baseline_keys & current_keys)
    missing_keys = sorted(baseline_keys - current_keys)
    added_keys = sorted(current_keys - baseline_keys)

    class_changed: list[dict] = []
    status_changed: list[dict] = []
    confidence_changed: list[dict] = []
    max_delta = 0.0

    for key in common_keys:
        old = baseline[key]
        new = current[key]
        old_status = old.get("status", "ok")
        new_status = new.get("status", "ok")
        old_class = old.get("class_id", "")
        new_class = new.get("class_id", "")

        if old_status != new_status:
            status_changed.append({"key": key, "baseline_status": old_status, "current_status": new_status})
        if old_class != new_class:
            class_changed.append({"key": key, "baseline_class_id": old_class, "current_class_id": new_class})

        old_conf = parse_float(old.get("confidence"))
        new_conf = parse_float(new.get("confidence"))
        if old_conf is not None and new_conf is not None:
            delta = abs(new_conf - old_conf)
            max_delta = max(max_delta, delta)
            if delta > confidence_tolerance:
                confidence_changed.append(
                    {
                        "key": key,
                        "baseline_confidence": old_conf,
                        "current_confidence": new_conf,
                        "delta": delta,
                    }
                )

    return {
        "baseline_count": len(baseline),
        "current_count": len(current),
        "common_count": len(common_keys),
        "missing_count": len(missing_keys),
        "added_count": len(added_keys),
        "status_changed_count": len(status_changed),
        "class_changed_count": len(class_changed),
        "confidence_changed_count": len(confidence_changed),
        "max_confidence_delta": max_delta,
        "missing_keys": missing_keys,
        "added_keys": added_keys,
        "status_changed": status_changed,
        "class_changed": class_changed,
        "confidence_changed": confidence_changed,
    }


def write_comparison_markdown(summary: Mapping, output_md: Path) -> None:
    lines = [
        "# Prediction report comparison",
        "",
        "## Machine-readable summary",
        "",
        f"baseline_count={summary['baseline_count']}",
        f"current_count={summary['current_count']}",
        f"common_count={summary['common_count']}",
        f"missing_count={summary['missing_count']}",
        f"added_count={summary['added_count']}",
        f"status_changed_count={summary['status_changed_count']}",
        f"class_changed_count={summary['class_changed_count']}",
        f"confidence_changed_count={summary['confidence_changed_count']}",
        f"max_confidence_delta={summary['max_confidence_delta']:.6f}",
        "",
        "## Class changes",
        "",
        "| key | baseline_class_id | current_class_id |",
        "| --- | ---: | ---: |",
    ]
    for row in summary["class_changed"][:200]:
        lines.append(f"| {row['key']} | {row['baseline_class_id']} | {row['current_class_id']} |")

    lines.extend(["", "## Status changes", "", "| key | baseline_status | current_status |", "| --- | --- | --- |"])
    for row in summary["status_changed"][:200]:
        lines.append(f"| {row['key']} | {row['baseline_status']} | {row['current_status']} |")

    lines.extend(["", "## Largest confidence changes", "", "| key | baseline | current | delta |", "| --- | ---: | ---: | ---: |"])
    for row in sorted(summary["confidence_changed"], key=lambda item: item["delta"], reverse=True)[:200]:
        lines.append(f"| {row['key']} | {row['baseline_confidence']:.6f} | {row['current_confidence']:.6f} | {row['delta']:.6f} |")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two prediction CSV reports")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/classification_results/report_comparison.md"))
    parser.add_argument("--confidence-tolerance", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_prediction_records(args.baseline)
    current = load_prediction_records(args.current)
    summary = compare_records(baseline, current, confidence_tolerance=args.confidence_tolerance)
    write_comparison_markdown(summary, args.output)
    print(f"baseline_count={summary['baseline_count']}")
    print(f"current_count={summary['current_count']}")
    print(f"class_changed_count={summary['class_changed_count']}")
    print(f"status_changed_count={summary['status_changed_count']}")
    print(f"confidence_changed_count={summary['confidence_changed_count']}")
    print(f"max_confidence_delta={summary['max_confidence_delta']:.6f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
