"""Run the ToothSeg training pipeline from Google Colab.

The launcher clones a selected repository revision into the Colab runtime,
installs the training extra, normalizes the mounted dataset layout, and calls
the repository CLI. Dataset and output paths may point at /content or a
mounted Google Drive directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_DIR = Path("/content/toothseg-repo")
DEFAULT_OUTPUT_DIR = Path("/content/toothseg-output")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_repository(repo_url: str, repo_ref: str, repo_dir: Path) -> None:
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    if len(repo_ref) < 40:
        run(["git", "clone", "--depth", "1", "--branch", repo_ref, repo_url, str(repo_dir)])
    else:
        run(["git", "clone", "--filter", "blob:none", "--no-checkout", repo_url, str(repo_dir)])
        run(["git", "checkout", repo_ref], cwd=repo_dir)


def install_repository(repo_dir: Path) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[train]"], cwd=repo_dir)


def _prepared_layout(dataset_dir: Path) -> str | None:
    if all((dataset_dir / relative).exists() for relative in (
        "images/train", "images/val", "labels/train", "labels/val"
    )):
        return "images"
    if all((dataset_dir / relative).exists() for relative in (
        "train", "val", "labels/train", "labels/val"
    )):
        return "split"
    return None


def locate_prepared_dataset(dataset_dir: Path) -> tuple[Path, str] | None:
    """Find a prepared dataset at the supplied path or one level below it."""
    candidates = [dataset_dir]
    if dataset_dir.is_dir():
        candidates.extend(path for path in dataset_dir.iterdir() if path.is_dir())
    for candidate in candidates:
        layout = _prepared_layout(candidate)
        if layout:
            return candidate, layout
    return None


def prepare_dataset(dataset_dir: Path, repo_dir: Path, output_dir: Path, val_ratio: float, seed: int) -> Path:
    prepared_dir = output_dir / "dataset"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, str(repo_dir / "cli" / "main.py"), "prepare",
        "--data", str(dataset_dir), "--output", str(prepared_dir),
        "--val-ratio", str(val_ratio), "--seed", str(seed),
    ], cwd=repo_dir)
    return prepared_dir


def materialize_prepared_data_yaml(dataset_dir: Path, output_dir: Path, layout: str) -> Path:
    """Create a runtime YAML and a standard images/labels view when needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    view_root = dataset_dir
    if layout == "split":
        view_root = output_dir / "dataset-view"
        for relative in ("images/train", "images/val", "labels/train", "labels/val"):
            target = view_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source_relative = relative.replace("images/", "", 1) if relative.startswith("images/") else relative
            source = dataset_dir / source_relative
            if target.exists():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.copytree(source, target)

    source_yaml = dataset_dir / "data.yaml"
    if not source_yaml.is_file():
        source_yaml = view_root / "data.yaml"
    lines = source_yaml.read_text(encoding="utf-8").splitlines() if source_yaml.is_file() else [
        "train: images/train", "val: images/val", "names:", "  0: tooth"
    ]
    for index, line in enumerate(lines):
        if line.lstrip().startswith("path:"):
            lines[index] = f"path: {view_root.resolve().as_posix()}"
            break
    else:
        lines.insert(0, f"path: {view_root.resolve().as_posix()}")
    runtime_yaml = output_dir / "dataset.yaml"
    runtime_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return runtime_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=os.environ.get("TOOTHSEG_REPO_URL"), required=False)
    parser.add_argument("--repo-ref", default=os.environ.get("TOOTHSEG_REPO_REF", "main"))
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset path in /content or mounted Drive.")
    parser.add_argument("--dataset-mode", choices=("auto", "prepared", "raw"), default="auto")
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model name or checkpoint.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=None, help="Usually 0 on a Colab GPU; omit for automatic selection.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="tooth-detect-colab")
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

    located = locate_prepared_dataset(args.dataset)
    prepared = located is not None
    if args.dataset_mode == "prepared" and not prepared:
        raise SystemExit(
            "--dataset-mode prepared could not find a prepared dataset below "
            f"{args.dataset}. Expected images/train + labels/train, or train + labels/train."
        )
    if args.dataset_mode == "raw" or (args.dataset_mode == "auto" and not prepared):
        prepared_root = prepare_dataset(args.dataset, args.repo_dir, args.output_dir, args.val_ratio, args.seed)
        data_yaml = materialize_prepared_data_yaml(prepared_root, args.output_dir / ".runtime", "images")
    else:
        data_root, layout = located
        data_yaml = materialize_prepared_data_yaml(data_root, args.output_dir / ".runtime", layout)

    command = [
        sys.executable, str(args.repo_dir / "cli" / "main.py"), "train",
        "--data-yaml", str(data_yaml), "--model", args.model,
        "--epochs", str(args.epochs), "--imgsz", str(args.imgsz),
        "--batch", str(args.batch), "--project", str(args.output_dir / "train"),
        "--name", args.name,
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
