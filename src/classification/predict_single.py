from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.config import get_config, resolve_path
from classification.inference import load_model_for_inference, predict_image, resolve_inference_image_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one aluminum profile defect image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None, help="Checkpoint or classifier.torchscript.pt")
    parser.add_argument("--env", choices=["auto", "windows", "linux"], default="auto")
    parser.add_argument("--json", type=Path, default=None, help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.env)
    model_path = args.model or Path("outputs/classification_results/deploy/classifier.torchscript.pt")
    model_path = resolve_path(config, model_path)
    image_path = resolve_path(config, args.image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(model_path, device)
    image_size = resolve_inference_image_size(model_path, config.image_size)
    result = predict_image(model, image_path, image_size=image_size, device=device)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.json:
        output_path = resolve_path(config, args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
