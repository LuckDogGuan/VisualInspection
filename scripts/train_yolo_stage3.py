from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def read_last_metrics(results_csv: Path) -> dict[str, str]:
    with results_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {key.strip(): value for key, value in rows[-1].items()}


def write_report(run_dir: Path, metrics: dict[str, str]) -> None:
    metric_keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    lines = [
        "# YOLO Stage 3 Training Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Final Metrics",
        "",
        "```text",
    ]
    for key in metric_keys:
        if key in metrics:
            lines.append(f"{key}: {metrics[key]}")
    lines.extend(
        [
            "```",
            "",
            "## Key Files",
            "",
            "```text",
            f"best weights: {run_dir / 'weights' / 'best.pt'}",
            f"last weights: {run_dir / 'weights' / 'last.pt'}",
            f"results csv: {run_dir / 'results.csv'}",
            f"results plot: {run_dir / 'results.png'}",
            f"validation labels: {run_dir / 'val_batch0_labels.jpg'}",
            f"validation predictions: {run_dir / 'val_batch0_pred.jpg'}",
            "```",
            "",
            "Use the validation prediction images to check whether boxes are visually useful, not only the numeric metrics.",
        ]
    )
    (run_dir / "training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="outputs/yolo_stage3_manual")
    parser.add_argument("--name", default="server_train")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    model = YOLO(args.model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    run_dir = Path(result.save_dir)
    metrics = read_last_metrics(run_dir / "results.csv")
    write_report(run_dir, metrics)
    print(f"run_dir={run_dir}")
    print(run_dir / "training_report.md")


if __name__ == "__main__":
    main()
