"""Run the ToothSeg training pipeline from a Kaggle Notebook.

The script intentionally clones the repository at runtime so that the Kaggle
input only needs to contain the dataset and this launcher.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_DIR = Path("/kaggle/working/toothseg-repo")
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/toothseg-output")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_repository(repo_url: str, repo_ref: str, repo_dir: Path) -> None:
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    # Branches/tags can use a shallow clone. A commit hash needs a normal
    # object checkout because it is not accepted by clone's --branch option.
    if len(repo_ref) < 40:
        run(["git", "clone", "--depth", "1", "--branch", repo_ref, repo_url, str(repo_dir)])
    else:
        run(["git", "clone", "--filter", "blob:none", "--no-checkout", repo_url, str(repo_dir)])
        run(["git", "checkout", repo_ref], cwd=repo_dir)


def install_repository(repo_dir: Path) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[train]"], cwd=repo_dir)


def is_prepared_dataset(dataset_dir: Path) -> bool:
    return all(
        (dataset_dir / relative).exists()
        for relative in (
            "images/train",
            "images/val",
            "labels/train",
            "labels/val",
            "data.yaml",
        )
    )


def prepare_dataset(dataset_dir: Path, repo_dir: Path, output_dir: Path, val_ratio: float, seed: int) -> Path:
    prepared_dir = output_dir / "dataset"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(repo_dir / "cli" / "main.py"),
            "prepare",
            "--data",
            str(dataset_dir),
            "--output",
            str(prepared_dir),
            "--val-ratio",
            str(val_ratio),
            "--seed",
            str(seed),
        ],
        cwd=repo_dir,
    )
    return prepared_dir


def materialize_prepared_data_yaml(dataset_dir: Path, output_dir: Path) -> Path:
    """Make a Kaggle-local YAML whose path points at the mounted dataset."""
    source = dataset_dir / "data.yaml"
    if not source.is_file():
        raise SystemExit(f"Prepared dataset is missing data.yaml: {source}")
    lines = source.read_text(encoding="utf-8").splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("path:"):
            lines[index] = f"path: {dataset_dir.resolve().as_posix()}"
            replaced = True
            break
    if not replaced:
        lines.insert(0, f"path: {dataset_dir.resolve().as_posix()}")
    runtime_yaml = output_dir / "dataset.yaml"
    runtime_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return runtime_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=os.environ.get("TOOTHSEG_REPO_URL"), required=False)
    parser.add_argument("--repo-ref", default=os.environ.get("TOOTHSEG_REPO_REF", "main"))
    parser.add_argument("--dataset", type=Path, required=True, help="Mounted Kaggle dataset directory.")
    parser.add_argument("--dataset-mode", choices=("auto", "prepared", "raw"), default="auto")
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=None, help="Usually 0 on a Kaggle GPU; omit for Ultralytics default.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="tooth-detect-kaggle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.repo_url:
        raise SystemExit("--repo-url is required (or set TOOTHSEG_REPO_URL).")
    if not args.dataset.exists():
        raise SystemExit(f"Dataset directory does not exist: {args.dataset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clone_repository(args.repo_url, args.repo_ref, args.repo_dir)
    install_repository(args.repo_dir)

    prepared = is_prepared_dataset(args.dataset)
    if args.dataset_mode == "prepared" and not prepared:
        raise SystemExit("--dataset-mode prepared requires images/train, images/val, labels/train, labels/val and data.yaml.")
    if args.dataset_mode == "raw" or (args.dataset_mode == "auto" and not prepared):
        data_root = prepare_dataset(args.dataset, args.repo_dir, args.output_dir, args.val_ratio, args.seed)
        data_yaml = data_root / "data.yaml"
    else:
        data_root = args.dataset
        data_yaml = materialize_prepared_data_yaml(data_root, args.output_dir)

    command = [
        sys.executable,
        str(args.repo_dir / "cli" / "main.py"),
        "train",
        "--data-yaml",
        str(data_yaml),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--project",
        str(args.output_dir / "train"),
        "--name",
        args.name,
    ]
    if args.weights:
        command.extend(["--weights", str(args.weights)])
    if args.device:
        command.extend(["--device", args.device])
    if args.no_amp:
        command.append("--no-amp")
    run(command, cwd=args.repo_dir)
    print(f"Training finished. Results are under: {args.output_dir / 'train' / args.name}")


if __name__ == "__main__":
    main()
