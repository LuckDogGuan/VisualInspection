from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.config import get_config, resolve_path
from classification.data import LabelRow, build_label_rows, is_image
from classification.inference import load_checkpoint_model
from classification.visualize_defects import imwrite_unicode, load_text_font


CLASS_ID_TO_CN = {
    0: "正常",
    1: "不导电",
    2: "擦花",
    3: "横条压凹",
    4: "桔皮",
    5: "漏底",
    6: "碰伤",
    7: "起坑",
    8: "凸粉",
    9: "涂层开裂",
    10: "脏点",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM rough localization images for classifier results")
    parser.add_argument("--env", choices=["auto", "windows", "linux"], default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/classification_results/models/model_best.pth.tar"))
    parser.add_argument("--input", type=Path, default=None, help="Image file or folder. If omitted, sample from data-root.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classification_results/gradcam_samples"))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--target-class", type=int, default=None, help="Default uses predicted class.")
    parser.add_argument("--per-class", type=int, default=2, help="Sample count per class when input is omitted.")
    parser.add_argument("--limit", type=int, default=0, help="Limit images when input is a folder. 0 means no limit.")
    parser.add_argument("--recursive", action="store_true", help="Collect images recursively for folder input.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay strength.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    text = str(value)
    for old in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', " "]:
        text = text.replace(old, "_")
    return text


def collect_images(input_path: Path, recursive: bool, limit: int) -> list[Path]:
    if input_path.is_file() and is_image(input_path):
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input image or folder not found: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    images = [path for path in sorted(iterator, key=lambda item: str(item).lower()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if limit > 0:
        return images[:limit]
    return images


def sample_labeled_images(data_root: Path, per_class: int) -> list[tuple[Path, int | None]]:
    grouped: dict[int, list[LabelRow]] = defaultdict(list)
    for row in build_label_rows(data_root):
        grouped[row.label].append(row)
    sampled: list[tuple[Path, int | None]] = []
    for label in sorted(grouped):
        for row in grouped[label][:per_class]:
            sampled.append((row.image_path, row.label))
    return sampled


def get_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    last_conv: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No convolution layer found for Grad-CAM.")
    return last_conv


def prepare_image(image_path: Path, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    resized = TF.resize(image, [image_size + 16, image_size + 16])
    cropped = TF.center_crop(resized, [image_size, image_size])
    tensor = TF.to_tensor(cropped)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    display_rgb = np.asarray(cropped)
    return tensor.unsqueeze(0), display_rgb


class GradCamRunner:
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, target_class: int | None) -> tuple[np.ndarray, int, float, list[float]]:
        self.model.zero_grad(set_to_none=True)
        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        pred_class = int(probabilities.argmax().item())
        class_id = pred_class if target_class is None else int(target_class)
        score = logits[0, class_id]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook did not capture activations or gradients.")

        activations = self.activations[0]
        gradients = self.gradients[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)
        cam -= cam.min()
        if float(cam.max().item()) > 0:
            cam /= cam.max()
        return cam.cpu().numpy(), pred_class, float(probabilities[pred_class].item()), [float(value) for value in probabilities.detach().cpu().tolist()]


def draw_header(image_bgr: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    font = load_text_font(size=22)
    line_height = 30
    height = line_height * len(lines) + 10
    draw.rectangle((0, 0, pil_image.width, height), fill=(255, 255, 255))
    y = 5
    for line in lines:
        draw.text((10, y), line, font=font, fill=(0, 0, 0))
        y += line_height
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def overlay_heatmap(display_rgb: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    height, width = display_rgb.shape[:2]
    heatmap = cv2.resize(cam, (width, height))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(heatmap_bgr, alpha, base_bgr, 1.0 - alpha, 0)


def write_report(rows: Sequence[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "output_path",
        "true_class_id",
        "true_class_cn",
        "pred_class_id",
        "pred_class_cn",
        "confidence",
        "target_class_id",
        "target_class_cn",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: Sequence[dict], output_md: Path, checkpoint: Path, image_size: int) -> None:
    lines = [
        "# Grad-CAM 粗定位热力图结果",
        "",
        "## 说明",
        "",
        "本结果使用分类模型生成热力图，只表示模型大概关注区域，不等同于正式缺陷检测框。",
        "",
        "## 运行信息",
        "",
        f"- 检查点：`{checkpoint}`",
        f"- 输入尺寸：{image_size}",
        f"- 输出图片数：{len(rows)}",
        "",
        "## 使用建议",
        "",
        "1. 如果热力图大致覆盖缺陷区域，说明分类模型关注点有参考价值。",
        "2. 如果热力图经常关注背景、边缘或反光，说明需要改进拍摄条件或训练数据。",
        "3. 后续正式框选缺陷位置仍建议使用 YOLO 检测模型。",
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = get_config(args.env)
    checkpoint = resolve_path(config, args.checkpoint)
    output_dir = resolve_path(config, args.output_dir)
    data_root = resolve_path(config, args.data_root or config.classification_root)
    image_size = int(args.image_size or config.image_size)

    if checkpoint.name.endswith(".torchscript.pt"):
        raise ValueError("Grad-CAM requires a checkpoint such as model_best.pth.tar, not TorchScript.")

    if args.input is None:
        image_items = sample_labeled_images(data_root, per_class=args.per_class)
    else:
        input_path = resolve_path(config, args.input)
        image_items = [(path, None) for path in collect_images(input_path, recursive=args.recursive, limit=args.limit)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_model(checkpoint, device)
    target_layer = get_target_layer(model)
    runner = GradCamRunner(model, target_layer, device)

    rows: list[dict] = []
    try:
        for index, (image_path, true_class) in enumerate(image_items, start=1):
            tensor, display_rgb = prepare_image(image_path, image_size)
            cam, pred_class, confidence, _probabilities = runner.generate(tensor, target_class=args.target_class)
            target_class = pred_class if args.target_class is None else int(args.target_class)
            pred_cn = CLASS_ID_TO_CN.get(pred_class, f"类别{pred_class}")
            target_cn = CLASS_ID_TO_CN.get(target_class, f"类别{target_class}")
            true_cn = "" if true_class is None else CLASS_ID_TO_CN.get(true_class, f"类别{true_class}")

            folder = output_dir / "按预测类别" / f"{pred_class:02d}_{safe_name(pred_cn)}"
            output_name = f"{index:04d}__预测_{pred_class:02d}_{safe_name(pred_cn)}__置信度_{confidence:.3f}__{safe_name(image_path.stem)}.jpg"
            output_path = folder / output_name
            overlay = overlay_heatmap(display_rgb, cam, alpha=args.alpha)
            header_lines = [
                f"预测：{pred_cn}  置信度：{confidence:.3f}",
                f"热力图目标：{target_cn}",
            ]
            if true_cn:
                header_lines.insert(0, f"真实：{true_cn}")
            annotated = draw_header(overlay, header_lines)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not imwrite_unicode(output_path, annotated):
                raise RuntimeError(f"Failed to write output image: {output_path}")

            rows.append(
                {
                    "source_path": str(image_path),
                    "output_path": str(output_path),
                    "true_class_id": "" if true_class is None else true_class,
                    "true_class_cn": true_cn,
                    "pred_class_id": pred_class,
                    "pred_class_cn": pred_cn,
                    "confidence": f"{confidence:.6f}",
                    "target_class_id": target_class,
                    "target_class_cn": target_cn,
                }
            )
            if index % 50 == 0 or index == len(image_items):
                print(f"processed={index}/{len(image_items)}")
    finally:
        runner.close()

    write_report(rows, output_dir / "gradcam_report.csv")
    write_summary(rows, output_dir / "summary.md", checkpoint=checkpoint, image_size=image_size)
    print(f"gradcam_images={len(rows)}")
    print(f"output_dir={output_dir}")
    print(f"report={output_dir / 'gradcam_report.csv'}")
    print(f"summary={output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
