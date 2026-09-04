from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _require_ultralytics() -> Any:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is required for train/predict. Install with: pip install .[train]") from exc
    return YOLO


def train_yolo_detection(data_yaml: Path, epochs: int = 50, imgsz: int = 1024, project: str = "output/train") -> Path:
    YOLO = _require_ultralytics()
    model = YOLO("yolo11n.pt")
    result = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, project=project, name="tooth-detect")
    return Path(result.save_dir) / "weights" / "best.pt"


def validate_yolo_detection(weights: Path, data_yaml: Path, project: str = "output/val") -> Path:
    YOLO = _require_ultralytics()
    model = YOLO(str(weights))
    result = model.val(data=str(data_yaml), project=project, name="tooth-detect")
    save_dir = getattr(result, "save_dir", None)
    return Path(save_dir) if save_dir is not None else Path(project) / "tooth-detect"


def predict_image(weights: Path, image_path: Path, conf: float = 0.25) -> dict[str, Any]:
    YOLO = _require_ultralytics()
    model = YOLO(str(weights))
    result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    boxes = []
    if result.boxes is not None:
        for box in result.boxes.xyxy.tolist():
            x1, y1, x2, y2 = map(float, box)
            boxes.append([x1, y1, x2, y2])
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
    return {"image": image, "boxes": boxes, "raw": result}


def save_prediction(result: dict[str, Any], output_dir: Path, stem: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / f"{stem}_overlay.png"
    json_path = output_dir / f"{stem}_boxes.json"
    image = result["image"]
    image.save(overlay_path)
    payload = {"boxes": result["boxes"]}
    json_path.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"overlay": overlay_path, "json": json_path}
