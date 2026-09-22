from __future__ import annotations

from pathlib import Path

from .data import (
    export_pseudo_label_candidates,
    prepare_raw_yolo_dataset,
    publish_approved_pseudo_labels,
    validate_yolo_detection_dataset,
)
from .model import (
    predict_directory,
    predict_image,
    save_prediction,
    train_yolo_detection,
    validate_yolo_detection,
)


def prepare_dataset(
    raw_root: Path,
    output_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    fixed_val_manifest: Path | None = None,
) -> Path:
    return prepare_raw_yolo_dataset(
        raw_root, output_root, val_ratio=val_ratio, seed=seed, fixed_val_manifest=fixed_val_manifest
    )


def export_pseudo_labels(
    predict_root: Path,
    image_root: Path,
    candidate_root: Path,
    confidence_threshold: float = 0.9,
    teacher_weights: Path | None = None,
) -> Path:
    return export_pseudo_label_candidates(
        predict_root, image_root, candidate_root, confidence_threshold, teacher_weights
    )


def publish_pseudo_labels(candidate_root: Path, raw_root: Path) -> Path:
    return publish_approved_pseudo_labels(candidate_root, raw_root)


def train(
    data_yaml: Path,
    epochs: int = 50,
    imgsz: int = 1024,
    batch: int = -1,
    amp: bool = True,
    device: str | None = None,
    project: str = "output/train",
    name: str = "tooth-detect-v5",
) -> Path:
    return train_yolo_detection(
        data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        amp=amp,
        device=device,
        project=project,
        name=name,
    )


def validate(weights: Path, data_yaml: Path) -> Path:
    return validate_yolo_detection(weights, data_yaml)


def predict(weights: Path, image_path: Path, output_dir: Path, conf: float = 0.25) -> dict[str, Path]:
    result = predict_image(weights, image_path, conf=conf)
    return save_prediction(result, output_dir, image_path.stem)


def predict_dir(weights: Path, image_dir: Path, output_dir: Path, conf: float = 0.25) -> list[dict[str, Path]]:
    return predict_directory(weights, image_dir, output_dir, conf=conf)
