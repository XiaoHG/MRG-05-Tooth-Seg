from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetSummary:
    images: int
    labels: int
    classes: list[str]
    box_count: int
    min_boxes_per_image: int
    max_boxes_per_image: int


def _list_files(path: Path, suffixes: Iterable[str]) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes])


def _list_files_recursive(path: Path, suffixes: Iterable[str]) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes])


def _active_split_path(root: Path, kind: str) -> Path:
    split_root = root / kind
    if (split_root / "train").exists() or (split_root / "val").exists():
        return split_root
    return split_root


def load_classes(root: Path) -> list[str]:
    classes_path = root / "classes.txt"
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"Empty classes file: {classes_path}")
    return classes


def validate_yolo_detection_dataset(root: Path) -> DatasetSummary:
    images_dir = root / "images"
    labels_dir = root / "labels"
    classes = load_classes(root)

    if (images_dir / "train").exists() or (labels_dir / "train").exists():
        images = _list_files_recursive(images_dir / "train", {".jpg", ".jpeg", ".png", ".bmp", ".webp"}) + _list_files_recursive(
            images_dir / "val", {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        labels = _list_files_recursive(labels_dir / "train", {".txt"}) + _list_files_recursive(labels_dir / "val", {".txt"})
    else:
        images = _list_files_recursive(images_dir, {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        labels = _list_files_recursive(labels_dir, {".txt"})

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    if image_stems != label_stems:
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        raise ValueError(
            f"Image/label mismatch. missing_labels={missing_labels[:5]} missing_images={missing_images[:5]}"
        )

    box_count = 0
    per_image_counts: list[int] = []
    for label in labels:
        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        per_image_counts.append(len(lines))
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label line in {label.name}: {line}")
            cls, x, y, w, h = parts
            if cls not in {"0"}:
                raise ValueError(f"Unexpected class id in {label.name}: {cls}")
            vals = [float(x), float(y), float(w), float(h)]
            if any(v < 0 or v > 1 for v in vals):
                raise ValueError(f"Normalized coordinates out of range in {label.name}: {line}")
            box_count += 1

    return DatasetSummary(
        images=len(images),
        labels=len(labels),
        classes=classes,
        box_count=box_count,
        min_boxes_per_image=min(per_image_counts) if per_image_counts else 0,
        max_boxes_per_image=max(per_image_counts) if per_image_counts else 0,
    )


def prepare_yolo_detection_dataset(
    root: Path,
    output_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    summary = validate_yolo_detection_dataset(root)
    if summary.images == 0:
        raise ValueError("Dataset is empty")

    source_root = root / "raw" if (root / "raw").exists() else root
    source_images_dir = source_root / "images"
    source_labels_dir = source_root / "labels"

    if output_root.resolve() == root.resolve():
        raw_root = root / "raw"
        raw_images = raw_root / "images"
        raw_labels = raw_root / "labels"
        raw_root.mkdir(parents=True, exist_ok=True)

        if source_root == root:
            if (root / "images").exists():
                if not raw_images.exists():
                    shutil.move(str(root / "images"), str(raw_images))
                else:
                    shutil.rmtree(root / "images", ignore_errors=True)
            if (root / "labels").exists():
                if not raw_labels.exists():
                    shutil.move(str(root / "labels"), str(raw_labels))
                else:
                    shutil.rmtree(root / "labels", ignore_errors=True)
            source_images_dir = raw_images
            source_labels_dir = raw_labels

        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            target = root / sub
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
        split_root = root
    else:
        split_root = output_root
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            (split_root / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    images = _list_files(source_images_dir, {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    labels_dir = source_labels_dir

    stems = [p.stem for p in images]
    rng.shuffle(stems)
    val_count = max(1, round(len(stems) * val_ratio))
    val_stems = set(stems[:val_count])
    train_stems = [stem for stem in stems if stem not in val_stems]

    if not train_stems or not val_stems:
        raise ValueError("Split produced empty train or val set")

    for stem in train_stems:
        src_img = next(p for p in images if p.stem == stem)
        src_lbl = labels_dir / f"{stem}.txt"
        shutil.copy2(src_img, split_root / "images/train" / src_img.name)
        shutil.copy2(src_lbl, split_root / "labels/train" / src_lbl.name)

    for stem in val_stems:
        src_img = next(p for p in images if p.stem == stem)
        src_lbl = labels_dir / f"{stem}.txt"
        shutil.copy2(src_img, split_root / "images/val" / src_img.name)
        shutil.copy2(src_lbl, split_root / "labels/val" / src_lbl.name)

    data_yaml = split_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {split_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {summary.classes[0]}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "source": str(root),
        "output": str(split_root),
        "source_root": str(source_root),
        "seed": seed,
        "val_ratio": val_ratio,
        "train": len(train_stems),
        "val": len(val_stems),
    }
    (split_root / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return data_yaml
