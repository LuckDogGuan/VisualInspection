from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .config import PipelineConfig, resolve_path
from .data import DefectDataset, LabelRow, split_rows
from .modeling import build_classifier


class FixedRotation:
    def __init__(self, angles: Sequence[int]):
        self.angles = tuple(angles)

    def __call__(self, image):
        return image.rotate(random.choice(self.angles))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
            transforms.RandomHorizontalFlip(),
            FixedRotation((0, 90, 180, 270)),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item() * 100.0


def _build_loader(dataset, batch_size: int, shuffle: bool, workers: int, device: torch.device, prefetch_factor: int) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    train: bool,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
) -> tuple[float, float]:
    model.train(mode=train)
    total_loss = 0.0
    total_acc = 0.0
    total_items = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits.detach(), targets) * batch_size
        total_items += batch_size

    if total_items == 0:
        return 0.0, 0.0
    return total_loss / total_items, total_acc / total_items


def train_classifier(config: PipelineConfig, rows: Sequence[LabelRow], pretrained: bool = False) -> Path:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    seed_everything(config.seed)

    output_dir = resolve_path(config, config.output_dir)
    model_dir = resolve_path(config, config.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{config.architecture}_training_log.txt"

    train_rows, val_rows = split_rows(rows, val_ratio=config.val_ratio, seed=config.seed)
    if not train_rows or not val_rows:
        raise ValueError("Training and validation rows must both be non-empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    model = build_classifier(config.num_classes, architecture=config.architecture, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_loader = _build_loader(
        DefectDataset(train_rows, build_train_transforms(config.image_size)),
        batch_size=config.batch_size,
        shuffle=True,
        workers=config.workers,
        device=device,
        prefetch_factor=config.prefetch_factor,
    )
    val_loader = _build_loader(
        DefectDataset(val_rows, build_eval_transforms(config.image_size)),
        batch_size=config.batch_size * 2,
        shuffle=False,
        workers=config.workers,
        device=device,
        prefetch_factor=config.prefetch_factor,
    )

    best_acc = -1.0
    best_path = model_dir / "model_best.pth.tar"
    checkpoint_path = model_dir / "checkpoint.pth.tar"
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler, use_amp=use_amp)
            val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, train=False, scaler=None, use_amp=use_amp)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.4f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_acc": f"{val_acc:.4f}",
                }
            )
            handle.flush()
            state = {
                "epoch": epoch,
                "architecture": config.architecture,
                "num_classes": config.num_classes,
                "state_dict": model.state_dict(),
                "best_acc": max(best_acc, val_acc),
            }
            torch.save(state, checkpoint_path)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(state, best_path)
            print(f"epoch={epoch} train_acc={train_acc:.2f} val_acc={val_acc:.2f} val_loss={val_loss:.4f}")

    return best_path
