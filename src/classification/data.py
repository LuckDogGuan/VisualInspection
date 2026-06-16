from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from .config import CLASS_NAME_TO_ID, IGNORED_CLASS_FOLDERS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class LabelRow:
    image_path: Path
    label: int
    class_name: str


@dataclass(frozen=True)
class TestRow:
    image_path: Path


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def build_label_rows(classification_root: Path) -> List[LabelRow]:
    rows: List[LabelRow] = []
    if not classification_root.exists():
        raise FileNotFoundError(f"Classification root not found: {classification_root}")

    for class_dir in sorted(classification_root.iterdir(), key=lambda item: item.name.lower()):
        if not class_dir.is_dir():
            continue
        if class_dir.name in IGNORED_CLASS_FOLDERS:
            continue
        if class_dir.name not in CLASS_NAME_TO_ID:
            continue
        label = CLASS_NAME_TO_ID[class_dir.name]
        for image_path in sorted(class_dir.iterdir(), key=lambda item: item.name.lower()):
            if is_image(image_path):
                rows.append(LabelRow(image_path=image_path, label=label, class_name=class_dir.name))
    return rows


def build_test_rows(classification_root: Path, test_folder_name: str = "测试文件_未标注") -> List[TestRow]:
    test_dir = classification_root / test_folder_name
    if not test_dir.exists():
        return []
    return [TestRow(image_path=path) for path in sorted(test_dir.iterdir(), key=lambda item: item.name.lower()) if is_image(path)]


def write_label_csv(rows: Sequence[LabelRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["img_path", "label", "class_name"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"img_path": str(row.image_path), "label": row.label, "class_name": row.class_name})


def write_test_csv(rows: Sequence[TestRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["img_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"img_path": str(row.image_path)})


def split_rows(rows: Sequence[LabelRow], val_ratio: float, seed: int) -> Tuple[List[LabelRow], List[LabelRow]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    grouped: dict[int, list[LabelRow]] = defaultdict(list)
    for row in rows:
        grouped[row.label].append(row)

    rng = random.Random(seed)
    train_rows: List[LabelRow] = []
    val_rows: List[LabelRow] = []
    for label_rows in grouped.values():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        val_rows.extend(shuffled[:val_count])
        train_rows.extend(shuffled[val_count:])

    train_rows.sort(key=lambda row: str(row.image_path))
    val_rows.sort(key=lambda row: str(row.image_path))
    return train_rows, val_rows


def limit_rows_by_class(rows: Sequence[LabelRow], max_samples: int | None) -> List[LabelRow]:
    if max_samples is None or max_samples <= 0 or len(rows) <= max_samples:
        return list(rows)

    grouped: dict[int, list[LabelRow]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item.label, str(item.image_path))):
        grouped[row.label].append(row)

    limited: List[LabelRow] = []
    labels = sorted(grouped)
    offsets = {label: 0 for label in labels}
    while len(limited) < max_samples:
        added = False
        for label in labels:
            class_rows = grouped[label]
            offset = offsets[label]
            if offset < len(class_rows):
                limited.append(class_rows[offset])
                offsets[label] = offset + 1
                added = True
                if len(limited) == max_samples:
                    break
        if not added:
            break
    return limited


class DefectDataset(Dataset):
    def __init__(self, rows: Sequence[LabelRow], transform=None):
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(row.label, dtype=torch.long)


class ImagePathDataset(Dataset):
    def __init__(self, rows: Sequence[TestRow], transform=None):
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(row.image_path)
