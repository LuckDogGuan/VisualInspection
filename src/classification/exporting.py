from __future__ import annotations

import json
from io import BytesIO
import warnings
from pathlib import Path

import torch

from .config import CLASS_ID_TO_CN, CLASS_ID_TO_SUBMISSION
from .modeling import build_classifier


def export_torchscript_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    image_size: int = 384,
) -> tuple[Path, Path]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    architecture = checkpoint.get("architecture", "resnet50")
    num_classes = int(checkpoint.get("num_classes", 11))
    model = build_classifier(num_classes=num_classes, architecture=architecture, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    example_input = torch.zeros(1, 3, image_size, image_size)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.jit")
        scripted = torch.jit.trace(model, example_input)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "classifier.torchscript.pt"
    labels_path = output_dir / "labels.json"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.jit")
        buffer = BytesIO()
        torch.jit.save(scripted, buffer)
    artifact_path.write_bytes(buffer.getvalue())

    labels = {
        "image_size": image_size,
        "architecture": architecture,
        "num_classes": num_classes,
        "classes": [
            {
                "class_id": class_id,
                "class_name_cn": CLASS_ID_TO_CN[class_id],
                "submission_label": CLASS_ID_TO_SUBMISSION[class_id],
            }
            for class_id in range(num_classes)
        ],
    }
    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact_path, labels_path
