from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Conda's MKL runtime and PyTorch can load separate Intel OpenMP DLLs on Windows.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image, ImageDraw


def _require_ultralytics() -> Any:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is required for train/predict. Install with: pip install .[train]") from exc
    return YOLO


def train_yolo_detection(
    data_yaml: Path,
    epochs: int = 50,
    imgsz: int = 1024,
    batch: int = -1,
    amp: bool = True,
    device: str | None = None,
    project: str = "output/train",
    name: str = "tooth-detect-v5",
    weights: Path | None = None,
) -> Path:
    YOLO = _require_ultralytics()
    model = YOLO(str(weights) if weights is not None else "yolo11n.pt")
    project_path = Path(project).resolve()
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "amp": amp,
        "project": str(project_path),
        "name": name,
    }
    if device:
        kwargs["device"] = device
    result = model.train(**kwargs)
    return Path(result.save_dir) / "weights" / "best.pt"


def validate_yolo_detection(weights: Path, data_yaml: Path, project: str = "output/val") -> Path:
    YOLO = _require_ultralytics()
    model = YOLO(str(weights))
    result = model.val(data=str(data_yaml), project=project, name="tooth-detect")
    save_dir = getattr(result, "save_dir", None)
    return Path(save_dir) if save_dir is not None else Path(project) / "tooth-detect"


def _predict_image_with_model(
    model: Any, image_path: Path, conf: float = 0.25, device: str | None = None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"source": str(image_path), "conf": conf, "verbose": False}
    if device:
        kwargs["device"] = device
    result = model.predict(**kwargs)[0]

    source = Image.open(image_path).convert("RGB")
    image = source.copy()
    draw = ImageDraw.Draw(image)
    boxes = []
    confs = []
    if result.boxes is not None:
        for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            x1, y1, x2, y2 = map(float, box)
            boxes.append([x1, y1, x2, y2])
            confs.append(float(score))
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
    return {"image": image, "source": source, "boxes": boxes, "confs": confs, "raw": result}


def predict_image(
    weights: Path, image_path: Path, conf: float = 0.25, device: str | None = None
) -> dict[str, Any]:
    YOLO = _require_ultralytics()
    return _predict_image_with_model(YOLO(str(weights)), image_path, conf=conf, device=device)


def save_prediction(
    result: dict[str, Any],
    output_dir: Path,
    stem: str,
    source_path: Path | None = None,
) -> dict[str, Path]:
    sample_dir = output_dir / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = sample_dir / f"{stem}_overlay.png"
    original_path = sample_dir / f"{stem}_original.png"
    json_path = sample_dir / f"{stem}_detections.json"
    image = result["image"]
    source = result["source"]
    source.save(original_path)
    image.save(overlay_path)

    payload_boxes = []
    for idx, (box, score) in enumerate(zip(result["boxes"], result["confs"]), start=1):
        x1, y1, x2, y2 = box
        left = max(0, int(np.floor(x1)))
        top = max(0, int(np.floor(y1)))
        right = min(source.width, int(np.ceil(x2)))
        bottom = min(source.height, int(np.ceil(y2)))
        if right <= left or bottom <= top:
            continue
        crop = source.crop((left, top, right, bottom))
        crop_path = sample_dir / f"{stem}_{idx}.png"
        crop.save(crop_path)
        payload_boxes.append(
            {
                "index": idx,
                "confidence": score,
                "box_xyxy": [x1, y1, x2, y2],
                "crop": crop_path.name,
            }
        )

    payload = {"image": stem, "detections": payload_boxes}
    if source_path is not None:
        payload["source_image"] = source_path.as_posix()
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"original": original_path, "overlay": overlay_path, "json": json_path, "dir": sample_dir}


def predict_directory(
    weights: Path,
    image_dir: Path,
    output_dir: Path,
    conf: float = 0.25,
    device: str | None = None,
) -> list[dict[str, Path]]:
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in image_suffixes
    )
    if not image_paths:
        raise ValueError(f"No supported images found in: {image_dir}")

    outputs: list[dict[str, Path]] = []
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(image_dir)
        print(f"[{index}/{total}] {relative_path.as_posix()}")
        if device:
            result = predict_image(weights, image_path, conf=conf, device=device)
        else:
            result = predict_image(weights, image_path, conf=conf)
        outputs.append(
            save_prediction(
                result,
                output_dir,
                image_path.stem,
                source_path=relative_path,
            )
        )
    return outputs


def predict_directory_visualizations(
    weights: Path,
    image_dir: Path,
    output_dir: Path,
    conf: float = 0.25,
    device: str | None = None,
) -> Path:
    """Write one annotated image per input plus a single aggregate JSON result."""
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in image_suffixes
    )
    if not image_paths:
        raise ValueError(f"No supported images found in: {image_dir}")
    names = [path.name for path in image_paths]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(
            "Duplicate image names are not supported for aggregate predictions: "
            + ", ".join(duplicate_names)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    YOLO = _require_ultralytics()
    model = YOLO(str(weights))
    predictions: dict[str, dict[str, Any]] = {}
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(image_dir)
        print(f"[{index}/{total}] {relative_path.as_posix()}")
        if device:
            result = _predict_image_with_model(model, image_path, conf=conf, device=device)
        else:
            result = _predict_image_with_model(model, image_path, conf=conf)
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        result["image"].save(overlay_path)
        predictions[image_path.name] = {
            "source_image": relative_path.as_posix(),
            "visualization": overlay_path.name,
            "detections": [
                {
                    "index": detection_index,
                    "confidence": score,
                    "box_xyxy": box,
                }
                for detection_index, (box, score) in enumerate(zip(result["boxes"], result["confs"]), start=1)
            ],
        }

    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    return predictions_path
