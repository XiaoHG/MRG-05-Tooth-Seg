"""Model choices and backend readiness for comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    aliases: tuple[str, ...]
    family: str
    backend: str
    status: str
    note: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("yolo11n.pt", ("yolo11n", "yolo11n.pt"), "YOLO", "ultralytics", "ready", "Existing baseline."),
    ModelSpec("yolo11s.pt", ("yolo11s", "yolo11s.pt"), "YOLO", "ultralytics", "ready", "Existing comparison model."),
    ModelSpec("yolov8m.pt", ("yolov8m", "yolov8m.pt"), "YOLO", "ultralytics", "ready", "Ultralytics model."),
    ModelSpec("yolo26m.pt", ("yolo26m", "yolo26m.pt"), "YOLO", "ultralytics", "ready", "Ultralytics model; requires a recent Ultralytics release."),
    ModelSpec("rtdetr-l.pt", ("rtdetr-l", "rtdetr_l", "rtdetr-l.pt"), "RT-DETR", "ultralytics", "ready", "Existing RT-DETR baseline."),
    ModelSpec("fasterrcnn_resnet50_fpn", ("fasterrcnn", "faster-rcnn", "faster r-cnn", "fasterrcnn_resnet50_fpn"), "Faster R-CNN", "torchvision", "ready", "Native torchvision detection adapter."),
    ModelSpec("retinanet_resnet50_fpn", ("retinanet", "retinanet_resnet50_fpn"), "RetinaNet", "torchvision", "ready", "Native torchvision detection adapter."),
    ModelSpec("rtdetrv2-l", ("rtdetrv2-l", "rt-detrv2-l", "rtdetrv2_l"), "RT-DETRv2", "transformers", "ready", "Native Transformers adapter; requires the RT-DETRv2 checkpoint/package."),
    ModelSpec("rf-detr-m", ("rf-detr-m", "rf_detr_m", "rfdetr-m"), "RF-DETR", "rfdetr", "ready", "Native RF-DETR adapter; requires the rfdetr package."),
)


def _key(value: str) -> str:
    return Path(value).name.lower().strip()


def find_model_spec(value: str) -> ModelSpec | None:
    key = _key(value)
    for spec in MODEL_SPECS:
        if key in spec.aliases:
            return spec
    return None


def require_trainable_model(value: str) -> ModelSpec:
    spec = find_model_spec(value)
    if spec is None:
        return ModelSpec(value, (value,), "custom", "ultralytics", "custom", "Custom Ultralytics model or checkpoint.")
    if spec.status != "ready":
        raise ValueError(
            f"Model '{value}' is catalogued as {spec.status}, not ready for the current CLI. "
            f"Backend: {spec.backend}. {spec.note}"
        )
    return spec


def format_model_catalog() -> str:
    lines = ["name\tfamily\tbackend\tstatus\tnote"]
    lines.extend(
        f"{spec.name}\t{spec.family}\t{spec.backend}\t{spec.status}\t{spec.note}"
        for spec in MODEL_SPECS
    )
    return "\n".join(lines)
