from __future__ import annotations

import csv
import json
from io import BytesIO
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from PIL import Image

from .config import CLASS_ID_TO_CN, CLASS_ID_TO_SUBMISSION
from .modeling import build_classifier
from .train import build_eval_transforms


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint.get("architecture", "resnet50")
    num_classes = int(checkpoint.get("num_classes", 11))
    model = build_classifier(num_classes=num_classes, architecture=architecture, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_model_for_inference(model_path: Path, device: torch.device):
    if model_path.name.endswith(".torchscript.pt"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.jit")
            model = torch.jit.load(BytesIO(model_path.read_bytes()), map_location=device)
        model.eval()
        return model
    return load_checkpoint_model(model_path, device)


def resolve_inference_image_size(model_path: Path, default_image_size: int) -> int:
    labels_path = model_path.parent / "labels.json"
    if not labels_path.exists():
        return default_image_size
    try:
        metadata = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_image_size
    image_size = metadata.get("image_size", default_image_size)
    return int(image_size)


def predict_image(model: nn.Module, image_path: Path, image_size: int, device: torch.device) -> dict:
    transform = build_eval_transforms(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    class_id = int(probabilities.argmax().item())
    confidence = float(probabilities[class_id].item())
    return {
        "image_path": str(image_path),
        "class_id": class_id,
        "class_name_cn": CLASS_ID_TO_CN[class_id],
        "submission_label": CLASS_ID_TO_SUBMISSION[class_id],
        "confidence": confidence,
        "probabilities": [float(value) for value in probabilities.tolist()],
    }


def write_predictions_csv(predictions: Sequence[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "class_id", "class_name_cn", "submission_label", "confidence", "probabilities"]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            row = dict(prediction)
            row["probabilities"] = json.dumps(row["probabilities"], ensure_ascii=False)
            writer.writerow(row)
