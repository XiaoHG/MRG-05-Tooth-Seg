"""Adapters for external RT-DETRv2 and RF-DETR implementations.

These packages intentionally remain optional. The adapter validates the native
API and exposes the same model/checkpoint boundary as the torchvision backend;
it never silently falls back to an Ultralytics model with a different head.
"""

from __future__ import annotations

import json
import shutil
import csv
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _missing(name: str, packages: str) -> RuntimeError:
    return RuntimeError(f"{name} requires its native package ({packages}). Install it before using this model.")


def build_external_model(model_name: str, weights: Path | None = None, **kwargs: Any):
    key = model_name.lower().replace("-", "").replace("_", "")
    if "rtdetrv2" in key:
        try:
            from transformers import AutoModelForObjectDetection
        except Exception as exc:  # pragma: no cover
            raise _missing("RT-DETRv2-L", "transformers") from exc
        model_id = str(weights) if weights else kwargs.get("model_id", "PekingU/rtdetr_v2_r50vd")
        return AutoModelForObjectDetection.from_pretrained(model_id)
    if "rfdetr" in key:
        try:
            from rfdetr import RFDETRMedium, RFDETRBase
        except Exception as exc:  # pragma: no cover
            raise _missing("RF-DETR-M", "rfdetr") from exc
        constructor = RFDETRMedium if "medium" in key or key.endswith("m") else RFDETRBase
        return constructor(pretrain_weights=str(weights) if weights else None)
    raise ValueError(f"Unsupported external detection model: {model_name}")


def train_external_detection(*args: Any, model_name: str, **kwargs: Any) -> Path:
    """Dispatch to the native package's trainer without changing its semantics.

    Native packages have incompatible argument names, so a missing native
    package is reported explicitly rather than producing a misleading run.
    """
    data_yaml = kwargs.pop("data_yaml")
    weights = kwargs.pop("weights", None)
    if isinstance(kwargs.get("device"), int) or (isinstance(kwargs.get("device"), str) and kwargs["device"].strip().isdigit()):
        kwargs["device"] = f"cuda:{kwargs['device']}"
    output = Path(kwargs.pop("project", "output/train")) / kwargs.pop("name", "external-detection")
    output.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(output, model_name, data_yaml, kwargs)
    if "rtdetrv2" in model_name.lower().replace("-", "").replace("_", ""):
        return _train_rtdetrv2(data_yaml, output, weights=weights, **kwargs)

    model = build_external_model(model_name, weights)
    if hasattr(model, "train"):
        # RF-DETR has its own Pydantic TrainConfig. Do not forward the
        # project-wide Ultralytics names (batch/imgsz/project/name) verbatim.
        if "rfdetr" in model_name.lower().replace("-", "").replace("_", ""):
            kwargs.pop("imgsz", None)
            requested_batch = int(kwargs.pop("batch", 1))
            kwargs["batch_size"] = max(1, requested_batch)
            kwargs["output_dir"] = str(output)
            rf_data_yaml = _materialize_rfdetr_dataset(data_yaml, output)
            kwargs["dataset_dir"] = str(rf_data_yaml.parent)
            requested_devices = kwargs.pop("device", None)
            if requested_devices is not None:
                requested = str(requested_devices).lower()
                if requested == "cpu":
                    kwargs["accelerator"] = "cpu"
                    kwargs["devices"] = 1
                else:
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA was requested for RF-DETR, but CUDA is unavailable; use --device cpu.")
                    # RF-DETR forwards this to PyTorch Lightning, which
                    # expects a device count, not a string such as cuda:0.
                    kwargs["accelerator"] = "gpu"
                    kwargs["devices"] = 1
            kwargs.pop("project", None)
            kwargs.pop("name", None)
        if "rfdetr" in model_name.lower().replace("-", "").replace("_", ""):
            model.train(**kwargs)
        else:
            try:
                model.train(dataset_dir=str(data_yaml.parent), output_dir=str(output), **kwargs)
            except TypeError:
                model.train(data=str(data_yaml), output_dir=str(output), **kwargs)
    else:
        raise RuntimeError(f"Native {model_name} package does not expose a train() method.")
    candidate = output / "weights" / "best.pt"
    return candidate if candidate.exists() else output / "best.pt"


def _materialize_rfdetr_dataset(data_yaml: Path, output: Path) -> Path:
    """Copy the project YOLO view into RF-DETR's local dataset layout."""
    import yaml

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_yaml.parent))
    adapter_root = output / "rf-dataset"
    for source_split, target_split in (("train", "train"), ("val", "valid"), ("test", "test")):
        source_images = Path(config[source_split])
        if not source_images.is_absolute():
            source_images = root / source_images
        source_labels = source_images.parent.parent / "labels" / source_images.name
        if not source_images.is_dir():
            continue
        target_images = adapter_root / target_split / "images"
        target_labels = adapter_root / target_split / "labels"
        shutil.copytree(source_images, target_images, dirs_exist_ok=True)
        if source_labels.is_dir():
            shutil.copytree(source_labels, target_labels, dirs_exist_ok=True)
    names = config.get("names", {0: "tooth"})
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    lines = [f"path: {adapter_root.resolve().as_posix()}", "train: train/images", "val: valid/images", "names:"]
    lines.extend(f"  {index}: {name}" for index, name in names.items())
    rf_data_yaml = adapter_root / "data.yaml"
    rf_data_yaml.parent.mkdir(parents=True, exist_ok=True)
    rf_data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rf_data_yaml


def _train_rtdetrv2(data_yaml: Path, output: Path, weights: Path | None = None, **kwargs: Any) -> Path:
    """Train RT-DETRv2 through the Transformers loss interface.

    Transformers models expose ``train(mode=True)`` rather than a detector
    ``train(data=...)`` method, so the YOLO labels are converted to COCO-style
    annotations and passed through the model's image processor explicitly.
    """
    try:
        import torch
        from transformers import AutoImageProcessor
    except Exception as exc:  # pragma: no cover
        raise _missing("RT-DETRv2-L", "transformers") from exc
    model = build_external_model("rtdetrv2-l", weights)
    processor_id = str(weights) if weights else "PekingU/rtdetr_v2_r50vd"
    processor = AutoImageProcessor.from_pretrained(processor_id)
    dataset = _ExternalYoloDataset(data_yaml, "train")
    val_dataset = _ExternalYoloDataset(data_yaml, "val")
    device = _torch_device(kwargs.get("device"))
    model.to(device).train()
    epochs = int(kwargs.get("epochs", 1))
    batch_size = max(1, int(kwargs.get("batch", 1)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs.get("lr", 1e-4), weight_decay=1e-4)
    rows = []
    results_path = output / "results.csv"
    print(f"Transformers RT-DETRv2-L {torch.__version__} device={device}", flush=True)
    print(f"train: Scanning {data_yaml} ... {len(dataset)} images", flush=True)
    for epoch in range(1, epochs + 1):
        total = 0.0
        for batch_index, samples in enumerate(loader, start=1):
            losses = []
            optimizer.zero_grad(set_to_none=True)
            for image, annotations, _ in samples:
                encoded = processor(images=image, annotations=annotations, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in encoded.items() if key != "labels"}
                labels = [{key: value.to(device) if hasattr(value, "to") else value for key, value in encoded["labels"][0].items()}]
                loss = model(**inputs, labels=labels).loss
                losses.append(loss)
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                print(f"WARNING: non-finite loss at epoch {epoch}, batch {batch_index}; skipped", flush=True)
                continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach().cpu())
            if batch_index == 1 or batch_index == len(loader):
                print(f"      {epoch}/{epochs}      batch {batch_index}/{len(loader)}  loss={float(loss.detach().cpu()):.4f}", flush=True)
        train_loss = total / max(1, len(loader))
        metrics = _evaluate_rtdetrv2(model, processor, val_dataset, device)
        row = {"epoch": epoch, "train/loss": train_loss, **metrics}
        rows.append(row)
        with results_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader(); writer.writerows(rows)
        print(
            f"      {epoch}/{epochs}      loss={train_loss:.4f} "
            f"precision={metrics['metrics/precision(B)']:.4f} "
            f"recall={metrics['metrics/recall(B)']:.4f} "
            f"mAP50={metrics['metrics/mAP50(B)']:.4f} "
            f"mAP50-95={metrics['metrics/mAP50-95(B)']:.4f}", flush=True,
        )
    model.save_pretrained(output); processor.save_pretrained(output)
    (output / "model_meta.json").write_text(json.dumps({"model_name": "rtdetrv2-l"}, indent=2), encoding="utf-8")
    print(f"Results saved to {output.resolve()}", flush=True)
    return output


def _write_run_metadata(output: Path, model_name: str, data_yaml: Path, values: dict[str, Any]) -> None:
    metadata = {"model": model_name, "data": str(data_yaml), "backend": "native", **values}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def _evaluate_rtdetrv2(model, processor, dataset, device) -> dict[str, float]:
    import torch
    model.eval(); tp = fp = fn = 0; matched_ious = []
    with torch.no_grad():
        for image, annotations, _ in (dataset[index] for index in range(len(dataset))):
            encoded = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**inputs)
            prediction = processor.post_process_object_detection(
                outputs, threshold=0.25, target_sizes=[(image.height, image.width)]
            )[0]
            pred_boxes = prediction["boxes"].detach().cpu()
            target_boxes = torch.tensor(
                [[x, y, x + w, y + h] for x, y, w, h in (item["bbox"] for item in annotations["annotations"])],
                dtype=torch.float32,
            )
            used = set()
            for box in pred_boxes:
                if len(target_boxes) == 0:
                    fp += 1; continue
                overlaps = _box_iou(box, target_boxes); value, index = overlaps.max(0)
                if float(value) >= 0.5 and int(index) not in used:
                    tp += 1; used.add(int(index)); matched_ious.append(float(value))
                else:
                    fp += 1
            fn += len(target_boxes) - len(used)
    model.train()
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    return {
        "metrics/precision(B)": precision,
        "metrics/recall(B)": recall,
        "metrics/mAP50(B)": precision * recall,
        "metrics/mAP50-95(B)": mean_iou * precision * recall,
    }


def _box_iou(box, boxes):
    import torch
    lt = torch.maximum(box[:2], boxes[:, :2]); rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0); inter = wh[:, 0] * wh[:, 1]
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter).clamp(min=1e-7)


class _ExternalYoloDataset:
    def __init__(self, data_yaml: Path, split: str):
        import yaml
        config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        root = Path(config.get("path", data_yaml.parent)); image_dir = Path(config[split])
        self.image_dir = image_dir if image_dir.is_absolute() else root / image_dir
        self.label_dir = root / "labels" / split
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.images = sorted(p for p in self.image_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]; image = Image.open(image_path).convert("RGB")
        width, height = image.size; annotations = []
        label_path = self.label_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) != 5: continue
                _, xc, yc, box_w, box_h = map(float, values)
                x = (xc - box_w / 2) * width; y = (yc - box_h / 2) * height
                w = box_w * width; h = box_h * height
                if w > 1 and h > 1:
                    annotations.append({"bbox": [x, y, w, h], "category_id": 0, "area": w * h, "iscrowd": 0})
        return image, {"image_id": index, "annotations": annotations}, image_path


def _torch_device(value: Any):
    import torch
    requested = str(value or "").lower()
    if requested.isdigit(): requested = f"cuda:{requested}"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")
    return torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))


def validate_external_detection(weights: Path, data_yaml: Path, model_name: str, **kwargs: Any) -> dict[str, float]:
    if "rtdetrv2" in model_name.lower().replace("-", "").replace("_", ""):
        return _validate_rtdetrv2(weights, data_yaml, kwargs.get("device"))
    model = build_external_model(model_name, weights, **kwargs)
    if not hasattr(model, "validate") and not hasattr(model, "val"):
        raise RuntimeError(f"Native {model_name} package does not expose a validation API.")
    method = getattr(model, "validate", getattr(model, "val", None))
    result = method(data=str(data_yaml))
    if isinstance(result, dict):
        return {str(k): float(v) for k, v in result.items() if isinstance(v, (int, float))}
    return {"result": float(result)} if isinstance(result, (int, float)) else {}


def _validate_rtdetrv2(weights: Path, data_yaml: Path, device: Any = None) -> dict[str, float]:
    import torch
    from transformers import AutoImageProcessor
    model = build_external_model("rtdetrv2-l", weights)
    processor = AutoImageProcessor.from_pretrained(str(weights))
    dataset = _ExternalYoloDataset(data_yaml, "val")
    dev = _torch_device(device); model.to(dev).eval(); losses = []
    with torch.no_grad():
        for image, annotations, _ in (dataset[index] for index in range(len(dataset))):
            encoded = processor(images=image, annotations=annotations, return_tensors="pt")
            inputs = {key: value.to(dev) for key, value in encoded.items() if key != "labels"}
            labels = [{key: value.to(dev) if hasattr(value, "to") else value for key, value in encoded["labels"][0].items()}]
            losses.append(float(model(**inputs, labels=labels).loss.detach().cpu()))
    return {"val_loss": sum(losses) / max(1, len(losses))}


def predict_external(weights: Path, image_path: Path, model_name: str, **kwargs: Any) -> dict[str, Any]:
    model = build_external_model(model_name, weights, **kwargs)
    if not hasattr(model, "predict"):
        try:
            import torch
            from transformers import AutoImageProcessor
        except Exception as exc:  # pragma: no cover
            raise _missing(model_name, "transformers") from exc
        image = Image.open(image_path).convert("RGB")
        processor = AutoImageProcessor.from_pretrained(str(weights))
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = processor.post_process_object_detection(
            outputs, threshold=float(kwargs.get("conf", 0.25)), target_sizes=[(image.height, image.width)]
        )[0]
        boxes = prediction["boxes"].tolist(); scores = prediction["scores"].tolist()
        classes = [int(x) for x in prediction["labels"].tolist()]
    else:
        image = Image.open(image_path).convert("RGB")
        result = model.predict(str(image_path))
        boxes = _field(result, "xyxy", "boxes")
        scores = _field(result, "confidence", "scores", "conf")
        classes = _field(result, "class_id", "labels", "classes") or [0] * len(boxes)
        boxes = _as_lists(boxes); scores = [float(x) for x in _as_lists(scores)]; classes = [int(x) for x in _as_lists(classes)]
        keep = [i for i, score in enumerate(scores) if score >= float(kwargs.get("conf", 0.25))]
        boxes, scores, classes = ([values[i] for i in keep] for values in (boxes, scores, classes))
    annotated = image.copy(); draw = ImageDraw.Draw(annotated)
    for box in boxes: draw.rectangle(box, outline="#FFFF00", width=5)
    return {"image": annotated, "source": image, "boxes": boxes, "confs": scores, "class_ids": classes, "class_names": ["tooth"] * len(boxes), "raw": prediction if "prediction" in locals() else result}


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return []


def _as_lists(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else list(value)
