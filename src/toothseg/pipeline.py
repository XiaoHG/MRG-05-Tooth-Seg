from __future__ import annotations

from pathlib import Path

from .data import prepare_yolo_detection_dataset, validate_yolo_detection_dataset
from .model import predict_image, save_prediction, train_yolo_detection, validate_yolo_detection


def prepare_dataset(input_root: Path, output_root: Path, val_ratio: float = 0.2, seed: int = 42) -> Path:
    return prepare_yolo_detection_dataset(input_root, output_root, val_ratio=val_ratio, seed=seed)


def train(data_yaml: Path, epochs: int = 50, imgsz: int = 1024) -> Path:
    return train_yolo_detection(data_yaml, epochs=epochs, imgsz=imgsz)


def validate(weights: Path, data_yaml: Path) -> Path:
    return validate_yolo_detection(weights, data_yaml)


def predict(weights: Path, image_path: Path, output_dir: Path, conf: float = 0.25) -> dict[str, Path]:
    result = predict_image(weights, image_path, conf=conf)
    return save_prediction(result, output_dir, image_path.stem)
