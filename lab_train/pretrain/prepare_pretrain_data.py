"""Build a deduplicated pretraining image manifest for DINOv3 continue-pretraining.

Self-supervised pretraining needs only images (labels are ignored). ``dataset/`` holds
one folder per public dataset (AlphaDent, COde-Dataset, Oral-Diseases, ...) with the
oral images dropped straight into the folder; datasets may overlap and a folder may
contain duplicates, so we deduplicate by SHA256 and emit one flat manifest that the
pretraining DataLoader consumes.

Output:
  <output>/pretrain_manifest.json
    {
      "base_dir": "...",          # lab_train package root
      "unique": N, "seen": M,     # unique images vs total files scanned
      "images": [{"path": "dataset/.../x.jpg", "sha256": "...", "size": 12345}]
    }

Image paths are stored relative to ``base_dir`` so the manifest stays valid after the
whole ``lab_train/`` directory is copied to the training server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # lab_train package root
DATASET_ROOT = ROOT / "dataset"
EXCLUDED_DIRS = {"pretrain"}  # holds the generated manifest, not source images
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CHUNK = 8 * 1024 * 1024


def default_image_dirs() -> list[Path]:
    """All top-level dirs under ``dataset/`` except the ``pretrain/`` manifest dir."""
    if not DATASET_ROOT.is_dir():
        return []
    return sorted(
        p for p in DATASET_ROOT.iterdir() if p.is_dir() and p.name not in EXCLUDED_DIRS
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _iter_images(image_dirs: list[Path]) -> list[Path]:
    seen_paths: set[Path] = set()
    out: list[Path] = []
    for directory in image_dirs:
        if not directory.is_dir():
            continue
        for p in sorted(directory.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                resolved = p.resolve()
                if resolved not in seen_paths:
                    seen_paths.add(resolved)
                    out.append(p)
    return out


def build_manifest(image_dirs: list[Path], base_dir: Path) -> dict:
    """Scan ``image_dirs``, deduplicate by SHA256, and return a manifest dict."""
    by_hash: dict[str, dict] = {}
    seen = 0
    for path in _iter_images(image_dirs):
        seen += 1
        digest = _sha256(path)
        if digest in by_hash:
            continue
        try:
            rel = path.resolve().relative_to(base_dir.resolve()).as_posix()
        except ValueError:
            rel = str(path.resolve())
        by_hash[digest] = {
            "path": rel,
            "sha256": digest,
            "size": path.stat().st_size,
        }

    images = sorted(by_hash.values(), key=lambda entry: entry["path"])
    return {
        "base_dir": str(base_dir.resolve()),
        "unique": len(images),
        "seen": seen,
        "images": images,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        default=None,
        help="Directories to scan recursively for images (deduplicated by SHA256). "
        "Defaults to all dataset-named dirs under dataset/ except pretrain/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset" / "pretrain" / "pretrain_manifest.json",
    )
    args = parser.parse_args()

    image_dirs = args.images or default_image_dirs()
    if not image_dirs:
        raise SystemExit(
            f"No dataset directories found under {DATASET_ROOT}. "
            "Place oral images in the dataset-named folders (or pass --images explicitly)."
        )

    manifest = build_manifest(image_dirs, ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Scanned {manifest['seen']} files across {len(image_dirs)} dir(s)")
    for d in image_dirs:
        try:
            label = d.resolve().relative_to(ROOT)
        except ValueError:  # --images may point outside the lab_train package
            label = d.resolve()
        print(f"  - {label}")
    print(f"Unique images after SHA256 dedup: {manifest['unique']}")
    print(f"Manifest written to: {args.output}")


if __name__ == "__main__":
    main()
