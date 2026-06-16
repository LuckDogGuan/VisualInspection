from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.config import get_config, resolve_path
from classification.exporting import export_torchscript_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained checkpoint to industrial-callable TorchScript artifact")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/classification_results/models/model_best.pth.tar"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classification_results/deploy"))
    parser.add_argument("--env", choices=["auto", "windows", "linux"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.env)
    artifact_path, labels_path = export_torchscript_checkpoint(
        resolve_path(config, args.checkpoint),
        resolve_path(config, args.output_dir),
        image_size=config.image_size,
    )
    print(f"artifact={artifact_path}")
    print(f"labels={labels_path}")


if __name__ == "__main__":
    main()
