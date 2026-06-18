from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    name: str
    output_dir: Path
    train_args: tuple[str, ...]


EXPERIMENTS = [
    Experiment("class_weight", Path("outputs/classification_results/exp_class_weight"), ("--class-weights",)),
    Experiment("weighted_sampler", Path("outputs/classification_results/exp_weighted_sampler"), ("--weighted-sampler",)),
    Experiment("focal", Path("outputs/classification_results/exp_focal"), ("--loss", "focal", "--focal-gamma", "2.0")),
    Experiment(
        "class_weight_focal",
        Path("outputs/classification_results/exp_class_weight_focal"),
        ("--class-weights", "--loss", "focal", "--focal-gamma", "2.0"),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classification optimization experiments sequentially")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--wait-pid", type=int, default=None, help="Wait for an already running first experiment")
    parser.add_argument("--only", action="append", default=None, help="Experiment name to run; repeatable")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    result = subprocess.run(["ps", "-p", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def wait_for_pid(pid: int, poll_seconds: int) -> None:
    log(f"waiting_for_pid={pid}")
    while pid_is_running(pid):
        log(f"pid_running={pid}; next_check_seconds={poll_seconds}")
        time.sleep(poll_seconds)
    log(f"pid_finished={pid}")


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"command_start={' '.join(command)}")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{timestamp()}] command_start={' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return_code = process.wait()
        handle.write(f"[{timestamp()}] command_exit={return_code}\n")
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")
    log(f"command_done={' '.join(command)}")


def deploy_model_path(output_dir: Path) -> Path:
    return output_dir / "deploy" / "classifier.torchscript.pt"


def run_training(args: argparse.Namespace, experiment: Experiment) -> None:
    model_path = deploy_model_path(experiment.output_dir)
    if args.skip_existing and model_path.exists():
        log(f"train_skip_existing={experiment.name} model={model_path}")
        return
    command = [
        args.python,
        "src/classification/run_pipeline.py",
        "--output-dir",
        str(experiment.output_dir),
        *experiment.train_args,
    ]
    run_command(command, experiment.output_dir / "queue_train.log")


def run_evaluation(args: argparse.Namespace, experiment: Experiment) -> None:
    model_path = deploy_model_path(experiment.output_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Deploy model not found: {model_path}")

    run_command(
        [
            args.python,
            "src/classification/screen_labeled_dataset.py",
            "--model",
            str(model_path),
            "--output-dir",
            str(experiment.output_dir / "full_screening"),
            "--no-copy",
        ],
        experiment.output_dir / "queue_labeled_eval.log",
    )
    run_command(
        [
            args.python,
            "src/classification/screen_raw_images.py",
            "--model",
            str(model_path),
            "--no-copy",
            "--report-dir",
            str(experiment.output_dir / "raw_images_quality"),
        ],
        experiment.output_dir / "queue_raw_eval.log",
    )

    baseline = Path("outputs/classification_results/raw_images_quality_baseline/raw_predictions.csv")
    current = experiment.output_dir / "raw_images_quality" / "raw_predictions.csv"
    if baseline.exists() and current.exists():
        run_command(
            [
                args.python,
                "src/classification/compare_prediction_reports.py",
                "--baseline",
                str(baseline),
                "--current",
                str(current),
                "--output",
                str(experiment.output_dir / "raw_images_quality_compare.md"),
            ],
            experiment.output_dir / "queue_raw_compare.log",
        )
    else:
        log(f"raw_compare_skip={experiment.name} baseline_exists={baseline.exists()} current_exists={current.exists()}")


def read_top1(summary_path: Path) -> str:
    if not summary_path.exists():
        return "missing"
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "Top1" in line and "%" in line:
            return line.strip()
    return "unknown"


def write_queue_summary(experiments: list[Experiment], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "model", "labeled_summary", "raw_summary", "raw_compare"])
        writer.writeheader()
        for experiment in experiments:
            writer.writerow(
                {
                    "experiment": experiment.name,
                    "model": str(deploy_model_path(experiment.output_dir)),
                    "labeled_summary": str(experiment.output_dir / "full_screening" / "summary.md"),
                    "raw_summary": str(experiment.output_dir / "raw_images_quality" / "summary.md"),
                    "raw_compare": str(experiment.output_dir / "raw_images_quality_compare.md"),
                }
            )


def main() -> None:
    args = parse_args()
    selected = [item for item in EXPERIMENTS if args.only is None or item.name in set(args.only)]
    if not selected:
        raise ValueError("No experiments selected")

    log(f"queue_start experiments={','.join(item.name for item in selected)}")
    if args.wait_pid is not None:
        wait_for_pid(args.wait_pid, args.poll_seconds)

    completed: list[Experiment] = []
    for experiment in selected:
        log(f"experiment_start={experiment.name}")
        run_training(args, experiment)
        run_evaluation(args, experiment)
        completed.append(experiment)
        write_queue_summary(completed, Path("outputs/classification_results/experiment_queue_summary.csv"))
        log(f"experiment_done={experiment.name}")

    log("queue_done")


if __name__ == "__main__":
    main()
