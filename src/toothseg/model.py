from __future__ import annotations

import json
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


def save_prediction(result: dict[str, Any], output_dir: Path, stem: str) -> dict[str, Path]:
    sample_dir = output_dir / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = sample_dir / f"{stem}_overlay.png"
    json_path = sample_dir / f"{stem}_detections.json"
    image = result["image"]
    source = result["source"]
    image.save(overlay_path)

    payload_boxes = []
    for idx, (box, score) in enumerate(zip(result["boxes"], result["confs"]), start=1):
        x1, y1, x2, y2 = box
        left = max(0, int(np.floor(x1)))
        top = max(0, int(np.floor(y1)))
        right = min(source.width, int(np.ceil(x2)))
        bottom = min(source.height, int(np.ceil(y2)))
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
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"overlay": overlay_path, "json": json_path, "dir": sample_dir}
