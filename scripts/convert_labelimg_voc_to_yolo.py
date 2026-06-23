from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_ORDER = ["dent", "powder", "stain", "crack", "transverse_bump"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_ORDER)}


def text_value(parent: ET.Element, path: str) -> str:
    value = parent.findtext(path)
    if value is None:
        raise ValueError(f"Missing XML field: {path}")
    return value.strip()


def convert_file(xml_path: Path, output_path: Path) -> tuple[int, list[str]]:
    root = ET.parse(xml_path).getroot()
    width = float(text_value(root, "size/width"))
    height = float(text_value(root, "size/height"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    lines: list[str] = []
    warnings: list[str] = []
    for obj in root.findall("object"):
        class_name = text_value(obj, "name")
        if class_name not in CLASS_TO_ID:
            warnings.append(f"{xml_path.name}: unsupported class {class_name}")
            continue

        xmin = float(text_value(obj, "bndbox/xmin"))
        ymin = float(text_value(obj, "bndbox/ymin"))
        xmax = float(text_value(obj, "bndbox/xmax"))
        ymax = float(text_value(obj, "bndbox/ymax"))

        xmin = max(0.0, min(width, xmin))
        xmax = max(0.0, min(width, xmax))
        ymin = max(0.0, min(height, ymin))
        ymax = max(0.0, min(height, ymax))
        if xmax <= xmin or ymax <= ymin:
            warnings.append(f"{xml_path.name}: invalid box {xmin},{ymin},{xmax},{ymax}")
            continue

        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        lines.append(
            f"{CLASS_TO_ID[class_name]} "
            f"{x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines), warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--project-root", default=str(Path.cwd()))
    args = parser.parse_args()

    project_root = Path(args.project_root)
    labels_dir = project_root / "data" / "yolo_stage3_manual" / "batches" / args.batch_id / "labels"
    if not labels_dir.exists():
        raise SystemExit(f"Labels directory does not exist: {labels_dir}")

    total_files = 0
    total_boxes = 0
    warnings: list[str] = []
    for xml_path in sorted(labels_dir.glob("*.xml")):
        output_path = xml_path.with_suffix(".txt")
        box_count, file_warnings = convert_file(xml_path, output_path)
        total_files += 1
        total_boxes += box_count
        warnings.extend(file_warnings)

    print(f"batch_id={args.batch_id}")
    print(f"xml_files={total_files}")
    print(f"boxes={total_boxes}")
    print(f"warnings={len(warnings)}")
    for warning in warnings[:50]:
        print(warning)


if __name__ == "__main__":
    main()
