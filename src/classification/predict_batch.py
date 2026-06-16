from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.config import get_config, resolve_path
from classification.data import is_image
from classification.inference import load_model_for_inference, predict_image, resolve_inference_image_size, write_predictions_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch predict aluminum profile defect images")
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder")
    parser.add_argument("--output", type=Path, default=Path("outputs/classification_results/predictions.csv"))
    parser.add_argument("--model", type=Path, default=None, help="Checkpoint or classifier.torchscript.pt")
    parser.add_argument("--env", choices=["auto", "windows", "linux"], default="auto")
    return parser.parse_args()


def collect_images(path: Path) -> list[Path]:
    if path.is_file() and is_image(path):
        return [path]
    if path.is_dir():
        return [item for item in sorted(path.iterdir(), key=lambda item: item.name.lower()) if is_image(item)]
    raise FileNotFoundError(f"No image file or folder found: {path}")


def main() -> None:
    args = parse_args()
    config = get_config(args.env)
    model_path = resolve_path(config, args.model or Path("outputs/classification_results/deploy/classifier.torchscript.pt"))
    input_path = resolve_path(config, args.input)
    output_path = resolve_path(config, args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(model_path, device)
    image_size = resolve_inference_image_size(model_path, config.image_size)
    predictions = [predict_image(model, image, image_size, device) for image in collect_images(input_path)]
    write_predictions_csv(predictions, output_path)
    print(f"predictions={len(predictions)} output={output_path}")


if __name__ == "__main__":
    main()
