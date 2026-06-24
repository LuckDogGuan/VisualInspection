from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO prediction and save image/txt results.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    print(Path(args.project) / args.name)


if __name__ == "__main__":
    main()
