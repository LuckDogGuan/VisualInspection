from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.config import get_config, resolve_path
from classification.data import build_label_rows, build_test_rows, limit_rows_by_class, write_label_csv, write_test_csv
from classification.exporting import export_torchscript_checkpoint
from classification.train import train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aluminum profile defect classification pipeline")
    parser.add_argument("--env", choices=["auto", "windows", "linux"], default="auto")
    parser.add_argument("--data-root", type=Path, default=None, help="Override data/ali2018")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Balanced training subset for smoke runs")
    parser.add_argument("--architecture", default=None, choices=["resnet18", "resnet50", "efficientnet_b0"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA AMP mixed precision")
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--class-weights", action="store_true", help="Use inverse-frequency class weights")
    parser.add_argument("--weighted-sampler", action="store_true", help="Sample rare classes more often during training")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma value when --loss focal is used")
    parser.add_argument("--index-only", action="store_true", help="Only write label/test CSV files")
    parser.add_argument("--dry-run", action="store_true", help="Scan data and print summary without training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.env)
    if args.data_root is not None:
        config = config.__class__(**{**config.__dict__, "classification_root": args.data_root})
    if args.epochs is not None:
        config = config.__class__(**{**config.__dict__, "epochs": args.epochs})
    if args.batch_size is not None:
        config = config.__class__(**{**config.__dict__, "batch_size": args.batch_size})
    if args.workers is not None:
        config = config.__class__(**{**config.__dict__, "workers": args.workers})
    if args.prefetch_factor is not None:
        config = config.__class__(**{**config.__dict__, "prefetch_factor": args.prefetch_factor})
    if args.image_size is not None:
        config = config.__class__(**{**config.__dict__, "image_size": args.image_size})
    if args.architecture is not None:
        config = config.__class__(**{**config.__dict__, "architecture": args.architecture})
    if args.output_dir is not None:
        config = config.__class__(
            **{
                **config.__dict__,
                "output_dir": args.output_dir,
                "model_dir": args.output_dir / "models",
            }
        )
    if args.no_amp:
        config = config.__class__(**{**config.__dict__, "use_amp": False})

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    data_root = resolve_path(config, config.classification_root)
    label_rows = build_label_rows(data_root)
    training_rows = label_rows
    if args.max_samples is not None:
        training_rows = limit_rows_by_class(label_rows, args.max_samples)
    test_rows = build_test_rows(data_root, config.test_folder_name)
    write_label_csv(label_rows, resolve_path(config, config.label_csv))
    write_test_csv(test_rows, resolve_path(config, config.test_csv))

    print(f"env={config.env} gpu_ids={config.gpu_ids}")
    print(
        "indexed_training_images="
        f"{len(label_rows)} active_training_images={len(training_rows)} test_images={len(test_rows)} "
        f"data_root={data_root} batch_size={config.batch_size} workers={config.workers} "
        f"amp={config.use_amp} prefetch_factor={config.prefetch_factor}"
    )
    print(
        "training_strategy="
        f"loss={args.loss} class_weights={args.class_weights} "
        f"weighted_sampler={args.weighted_sampler} focal_gamma={args.focal_gamma}"
    )
    if args.index_only or args.dry_run:
        return

    best_model = train_classifier(
        config,
        training_rows,
        pretrained=args.pretrained,
        loss_name=args.loss,
        use_class_weights=args.class_weights,
        use_weighted_sampler=args.weighted_sampler,
        focal_gamma=args.focal_gamma,
    )
    print(f"best_model={best_model}")
    artifact_path, labels_path = export_torchscript_checkpoint(
        best_model,
        resolve_path(config, config.output_dir / "deploy"),
        image_size=config.image_size,
    )
    print(f"deploy_artifact={artifact_path}")
    print(f"deploy_labels={labels_path}")


if __name__ == "__main__":
    main()
