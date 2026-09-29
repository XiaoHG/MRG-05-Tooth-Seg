from __future__ import annotations

import json
import hashlib
import math
import random
import shutil
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


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


def _find_duplicate_stems(paths: Iterable[Path]) -> dict[str, list[Path]]:
    by_stem: dict[str, list[Path]] = {}
    for path in paths:
        by_stem.setdefault(path.stem, []).append(path)
    return {stem: sorted(items) for stem, items in by_stem.items() if len(items) > 1}


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


def _validate_source_dataset(root: Path) -> tuple[list[Path], list[Path], list[str], int]:
    images = _list_files(root / "images", {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    labels = _list_files(root / "labels", {".txt"})
    classes = load_classes(root)
    image_by_stem = {path.stem: path for path in images}
    label_by_stem = {path.stem: path for path in labels}
    if set(image_by_stem) != set(label_by_stem):
        missing_labels = sorted(set(image_by_stem) - set(label_by_stem))
        missing_images = sorted(set(label_by_stem) - set(image_by_stem))
        raise ValueError(
            f"Image/label mismatch in {root}. "
            f"missing_labels={missing_labels[:5]} missing_images={missing_images[:5]}"
        )

    box_count = 0
    for label in labels:
        for line in label.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label line in {label}: {line}")
            cls, x, y, w, h = parts
            if cls != "0":
                raise ValueError(f"Unexpected class id in {label}: {cls}")
            vals = [float(x), float(y), float(w), float(h)]
            if any(v < 0 or v > 1 for v in vals):
                raise ValueError(f"Normalized coordinates out of range in {label}: {line}")
            if float(w) <= 0 or float(h) <= 0:
                raise ValueError(f"Non-positive box size in {label}: {line}")
            box_count += 1
    return images, labels, classes, box_count


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

    duplicate_images = _find_duplicate_stems(images)
    if duplicate_images:
        details = "; ".join(
            f"{stem}: {[path.name for path in paths]}" for stem, paths in sorted(duplicate_images.items())
        )
        raise ValueError(
            "Multiple image files share a YOLO label stem; move or rename the duplicates before training: "
            + details
        )

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


def prepare_raw_yolo_dataset(
    raw_root: Path,
    output_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    fixed_val_manifest: Path | None = None,
) -> Path:
    """Merge raw source directories into a reproducible train/val dataset."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    source_roots = sorted(
        path for path in raw_root.iterdir() if path.is_dir() and (path / "images").is_dir() and (path / "labels").is_dir()
    ) if raw_root.exists() else []
    if not source_roots:
        raise ValueError(f"No raw dataset sources found in: {raw_root}")

    records: list[tuple[str, Path, Path]] = []
    source_stats = []
    expected_classes: list[str] | None = None
    for source_root in source_roots:
        images, labels, classes, box_count = _validate_source_dataset(source_root)
        if expected_classes is None:
            expected_classes = classes
        elif classes != expected_classes:
            raise ValueError(f"Class definitions differ in {source_root}: {classes} != {expected_classes}")
        image_by_stem = {path.stem: path for path in images}
        label_by_stem = {path.stem: path for path in labels}
        for stem in image_by_stem:
            records.append((source_root.name, image_by_stem[stem], label_by_stem[stem]))
        source_stats.append({"source": source_root.name, "images": len(images), "labels": len(labels), "box_count": box_count})

    assert expected_classes is not None
    names = [image.name for _, image, _ in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image names exist across raw sources")
    duplicate_source_stems = _find_duplicate_stems(image for _, image, _ in records)
    if duplicate_source_stems:
        details = "; ".join(
            f"{stem}: {[f'{path.parent.parent.name}/{path.name}' for path in paths]}"
            for stem, paths in sorted(duplicate_source_stems.items())
        )
        raise ValueError(
            "Duplicate image stems exist across raw sources; resolve them before preparing the dataset: "
            + details
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            split_dir = output_root / kind / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)
    for cache_name in ("train.cache", "val.cache"):
        cache_path = output_root / "labels" / cache_name
        if cache_path.exists():
            cache_path.unlink()

    fixed_val_names: set[str] = set()
    if fixed_val_manifest is not None:
        payload = json.loads(fixed_val_manifest.read_text(encoding="utf-8"))
        fixed_val_names = set(payload.get("files", {}).get("val", []))
        if not fixed_val_names:
            raise ValueError(f"Fixed validation manifest has no val files: {fixed_val_manifest}")

    rng = random.Random(seed)
    assignments: dict[str, list[tuple[str, Path, Path]]] = {"train": [], "val": []}
    for source_name in sorted({source for source, _, _ in records}):
        source_records = [record for record in records if record[0] == source_name]
        if fixed_val_manifest is not None:
            fixed = [record for record in source_records if record[1].name in fixed_val_names]
            remaining = [record for record in source_records if record[1].name not in fixed_val_names]
            assignments["val"].extend(fixed)
            assignments["train"].extend(remaining)
        else:
            rng.shuffle(source_records)
            val_count = max(1, round(len(source_records) * val_ratio))
            if val_count >= len(source_records):
                val_count = len(source_records) - 1
            assignments["val"].extend(source_records[:val_count])
            assignments["train"].extend(source_records[val_count:])

    if fixed_val_manifest is not None:
        pseudo_records = [record for record in assignments["val"] if record[0] == "pseudo-label-v6"]
        if pseudo_records:
            raise ValueError("pseudo-label-v6 contains a fixed validation filename")
        if not assignments["val"]:
            raise ValueError("Fixed validation split is empty")

    for split, split_records in assignments.items():
        for source_name, image_path, label_path in split_records:
            target_name = image_path.name
            shutil.copy2(image_path, output_root / "images" / split / target_name)
            shutil.copy2(label_path, output_root / "labels" / split / f"{image_path.stem}.txt")

    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {expected_classes[0]}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "raw_root": str(raw_root),
        "output": str(output_root),
        "seed": seed,
        "val_ratio": val_ratio,
        "fixed_val_manifest": str(fixed_val_manifest) if fixed_val_manifest else None,
        "sources": source_stats,
        "train": len(assignments["train"]),
        "val": len(assignments["val"]),
        "classes": expected_classes,
    }
    (output_root / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_manifest = {
        **manifest,
        "files": {
            split: sorted(record[1].name for record in assignments[split])
            for split in ("train", "val")
        },
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "classes.txt").write_text("\n".join(expected_classes) + "\n", encoding="utf-8")
    (output_root / "notes.json").write_text(
        json.dumps(
            {
                "task": "single-class tooth detection",
                "label_format": "YOLO detection boxes",
                "raw_sources": [stat["source"] for stat in source_stats],
                "unlabeled_test_directory": "images/test",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return data_yaml


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_pseudo_label_candidates(
    predict_root: Path,
    candidate_root: Path,
    confidence_threshold: float = 0.9,
    teacher_weights: Path | None = None,
) -> Path:
    """Export high-confidence predictions for human review without touching raw data."""
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    json_files = sorted(predict_root.rglob("*_detections.json"))
    if not json_files:
        raise ValueError(f"No detection JSON files found in: {predict_root}")

    images_dir = candidate_root / "images"
    labels_dir = candidate_root / "labels"
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()

    records = []
    for json_path in json_files:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        stem = str(payload.get("image", json_path.name.removesuffix("_detections.json")))
        source_image = json_path.parent / f"{stem}_original.png"
        if not source_image.is_file():
            raise FileNotFoundError(f"Original prediction image not found: {source_image}")
        with Image.open(source_image) as image:
            width, height = image.size
        selected = []
        for detection in payload.get("detections", []):
            confidence = float(detection.get("confidence", 0))
            if not math.isfinite(confidence) or confidence < confidence_threshold:
                continue
            box = detection.get("box_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                continue
            x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
            y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            selected.append({"confidence": confidence, "box_xyxy": [x1, y1, x2, y2]})
        if not selected:
            continue
        candidate_image = images_dir / f"{stem}{source_image.suffix.lower()}"
        shutil.copy2(source_image, candidate_image)
        label_lines = []
        for detection in selected:
            x1, y1, x2, y2 = detection["box_xyxy"]
            label_lines.append("0 {:.8f} {:.8f} {:.8f} {:.8f}".format(
                ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height,
                (x2 - x1) / width, (y2 - y1) / height,
            ))
        (labels_dir / f"{candidate_image.stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        records.append({
            "image": candidate_image.name,
            "source_image": str(source_image),
            "source_image_sha256": _sha256(source_image),
            "source_detection_json": str(json_path),
            "candidate_boxes": selected,
            "candidate_count": len(selected),
            "review_status": "pending",
            "review_reason": None,
        })
    (candidate_root / "classes.txt").write_text("tooth\n", encoding="utf-8")
    manifest = {
        "version": "v6",
        "task": "single-class tooth detection pseudo-label candidates",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predict_root": str(predict_root),
        "teacher_weights": str(teacher_weights) if teacher_weights else None,
        "confidence_threshold": confidence_threshold,
        "review_required": True,
        "records": records,
    }
    manifest_path = candidate_root / "notes.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def publish_approved_pseudo_labels(candidate_root: Path, raw_root: Path) -> Path:
    """Publish only explicitly approved candidates into the immutable raw-source layout."""
    manifest_path = candidate_root / "notes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination = raw_root / "pseudo-label-v6"
    approved = [record for record in manifest.get("records", []) if record.get("review_status") == "approved"]
    if not approved:
        raise ValueError("No approved pseudo-label records found")
    (destination / "images").mkdir(parents=True, exist_ok=True)
    (destination / "labels").mkdir(parents=True, exist_ok=True)
    for record in approved:
        image_name = record["image"]
        shutil.copy2(candidate_root / "images" / image_name, destination / "images" / image_name)
        shutil.copy2(candidate_root / "labels" / f"{Path(image_name).stem}.txt", destination / "labels" / f"{Path(image_name).stem}.txt")
    (destination / "classes.txt").write_text("tooth\n", encoding="utf-8")
    published = {**manifest, "published_at": datetime.now(timezone.utc).isoformat(), "published_records": len(approved)}
    (destination / "notes.json").write_text(json.dumps(published, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


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
