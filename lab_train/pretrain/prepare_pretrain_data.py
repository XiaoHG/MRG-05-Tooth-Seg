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
      "corrupt": [...],           # skipped corrupt files (only with --validate-images)
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

from PIL import Image

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
            # ``._*`` files are macOS AppleDouble metadata, not real images.
            if (
                p.is_file()
                and p.suffix.lower() in IMAGE_SUFFIXES
                and not p.name.startswith("._")
            ):
                resolved = p.resolve()
                if resolved not in seen_paths:
                    seen_paths.add(resolved)
                    out.append(p)
    return out


def _relative_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:  # --images may point outside the lab_train package
        return str(path.resolve())


def _is_readable_image(path: Path) -> tuple[bool, str | None]:
    """Fully decode the image so truncated/corrupt files are detected."""
    try:
        with Image.open(path) as image:
            image.load()
        return True, None
    except Exception as exc:  # noqa: BLE001 - report whatever Pillow raises
        return False, f"{type(exc).__name__}: {exc}"


def build_manifest(
    image_dirs: list[Path],
    base_dir: Path,
    validate_images: bool = False,
) -> dict:
    """Scan ``image_dirs``, deduplicate by SHA256, and return a manifest dict.

    When ``validate_images`` is true every candidate is decoded first; truncated
    or corrupt files are listed under ``corrupt`` and excluded from ``images``.
    """
    by_hash: dict[str, dict] = {}
    corrupt: list[dict] = []
    seen = 0
    for path in _iter_images(image_dirs):
        seen += 1
        if validate_images:
            ok, error = _is_readable_image(path)
            if not ok:
                corrupt.append({"path": _relative_path(path, base_dir), "error": error})
                continue
        digest = _sha256(path)
        if digest in by_hash:
            continue
        by_hash[digest] = {
            "path": _relative_path(path, base_dir),
            "sha256": digest,
            "size": path.stat().st_size,
        }

    images = sorted(by_hash.values(), key=lambda entry: entry["path"])
    return {
        "base_dir": str(base_dir.resolve()),
        "unique": len(images),
        "seen": seen,
        "corrupt": corrupt,
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
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Decode every image and skip truncated/corrupt files (slower).",
    )
    args = parser.parse_args()

    image_dirs = args.images or default_image_dirs()
    if not image_dirs:
        raise SystemExit(
            f"No dataset directories found under {DATASET_ROOT}. "
            "Place oral images in the dataset-named folders (or pass --images explicitly)."
        )

    manifest = build_manifest(image_dirs, ROOT, validate_images=args.validate_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Scanned {manifest['seen']} files across {len(image_dirs)} dir(s)")
    for d in image_dirs:
        print(f"  - {_relative_path(d, ROOT)}")
    print(f"Unique images after SHA256 dedup: {manifest['unique']}")
    if manifest["corrupt"]:
        print(f"Skipped corrupt images: {len(manifest['corrupt'])}")
    print(f"Manifest written to: {args.output}")


if __name__ == "__main__":
    main()
