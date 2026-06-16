from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.config import CLASS_ID_TO_CN, CLASS_NAME_TO_ID, get_config, resolve_path
from classification.data import is_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO defect boxes on aluminum profile images")
    parser.add_argument("--weights", type=Path, default=Path("best.pt"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations"))
    parser.add_argument("--env", choices=["auto", "windows", "linux"], default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def collect_images(path: Path) -> list[Path]:
    if path.is_file() and is_image(path):
        return [path]
    if path.is_dir():
        return [item for item in sorted(path.iterdir(), key=lambda item: item.name.lower()) if is_image(item)]
    raise FileNotFoundError(f"No image file or folder found: {path}")


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image) -> bool:
    extension = path.suffix or ".jpg"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def load_text_font(size: int = 28):
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_box(image, xyxy, label: str):
    x1, y1, x2, y2 = [int(value) for value in xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    font = load_text_font()
    text_x = x1
    text_y = max(0, y1 - 34)
    text_bbox = draw.textbbox((text_x, text_y), label, font=font)
    draw.rectangle((text_bbox[0] - 3, text_bbox[1] - 2, text_bbox[2] + 3, text_bbox[3] + 2), fill=(255, 255, 255))
    draw.text((text_x, text_y), label, font=font, fill=(220, 0, 0))
    image[:, :, :] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def infer_label_from_parent(image_path: Path) -> str:
    parent = image_path.parent.name
    class_id = CLASS_NAME_TO_ID.get(parent)
    if class_id is None:
        return parent
    return CLASS_ID_TO_CN[class_id]


def main() -> None:
    args = parse_args()
    config = get_config(args.env)
    weights = resolve_path(config, args.weights)
    input_path = resolve_path(config, args.input)
    output_dir = resolve_path(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    count = 0
    for image_path in collect_images(input_path):
        image = imread_unicode(image_path)
        if image is None:
            print(f"skip unreadable image: {image_path}")
            continue
        results = model.predict(str(image_path), conf=args.conf, verbose=False)
        label = infer_label_from_parent(image_path)
        for result in results:
            for box in result.boxes:
                draw_box(image, box.xyxy[0].tolist(), label)
        output_path = output_dir / image_path.name
        if not imwrite_unicode(output_path, image):
            print(f"skip failed output image: {output_path}")
            continue
        count += 1
    print(f"visualized={count} output_dir={output_dir}")


if __name__ == "__main__":
    main()
