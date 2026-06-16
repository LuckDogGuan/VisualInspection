from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


CLASS_NAME_TO_ID: Dict[str, int] = {
    "Clean sample": 0,
    "non-conducting": 1,
    "scuffing": 2,
    "The transverse strip is dented": 3,
    "Orange peel": 4,
    "Drain bottom": 5,
    "Be injured by a collision": 6,
    "pitting": 7,
    "Convex powder": 8,
    "Coating cracking": 9,
    "Dirty spot": 10,
}

CLASS_ID_TO_CN: Dict[int, str] = {
    0: "正常/合格品",
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

CLASS_ID_TO_SUBMISSION = {
    0: "norm",
    1: "defect1",
    2: "defect2",
    3: "defect3",
    4: "defect4",
    5: "defect5",
    6: "defect6",
    7: "defect7",
    8: "defect8",
    9: "defect9",
    10: "defect10",
}

IGNORED_CLASS_FOLDERS = {"新增少样本缺陷", "测试文件_未标注"}

CLASSIFICATION_ROOT = Path("data") / "ali2018"
OUTPUT_DIR = Path("outputs") / "classification_results"

# One-click training defaults. Override by CLI only for smoke tests or experiments.
DEFAULT_GPU_IDS = "0"
DEFAULT_IMAGE_SIZE = 384
DEFAULT_SEED = 666
DEFAULT_VAL_RATIO = 0.12
DEFAULT_ARCHITECTURE = "resnet50"
DEFAULT_BATCH_SIZE = 96
DEFAULT_WINDOWS_WORKERS = 0
DEFAULT_LINUX_WORKERS = 16
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_USE_AMP = True
DEFAULT_PREFETCH_FACTOR = 4


@dataclass(frozen=True)
class PipelineConfig:
    env: str
    project_root: Path
    classification_root: Path
    test_folder_name: str
    output_dir: Path
    model_dir: Path
    label_csv: Path
    test_csv: Path
    gpu_ids: str
    image_size: int = DEFAULT_IMAGE_SIZE
    seed: int = DEFAULT_SEED
    val_ratio: float = DEFAULT_VAL_RATIO
    num_classes: int = 11
    architecture: str = DEFAULT_ARCHITECTURE
    batch_size: int = DEFAULT_BATCH_SIZE
    workers: int = DEFAULT_WINDOWS_WORKERS
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    use_amp: bool = DEFAULT_USE_AMP
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def detect_env() -> str:
    return "windows" if os.name == "nt" or sys.platform.startswith("win") else "linux"


def get_config(env: str = "auto", project_root: Path | None = None) -> PipelineConfig:
    root = project_root or get_project_root()
    env_name = env.lower()
    if env_name == "auto":
        env_name = detect_env()
    if env_name not in {"windows", "linux"}:
        raise ValueError(f"Unsupported env: {env}")

    classification_root = CLASSIFICATION_ROOT
    workers = DEFAULT_LINUX_WORKERS if env_name == "linux" else DEFAULT_WINDOWS_WORKERS

    return PipelineConfig(
        env=env_name,
        project_root=root,
        classification_root=classification_root,
        test_folder_name="测试文件_未标注",
        output_dir=OUTPUT_DIR,
        model_dir=OUTPUT_DIR / "models",
        label_csv=Path("data") / "label.csv",
        test_csv=Path("data") / "test.csv",
        gpu_ids=DEFAULT_GPU_IDS,
        workers=workers,
    )


def resolve_path(config: PipelineConfig, path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return config.project_root / candidate
