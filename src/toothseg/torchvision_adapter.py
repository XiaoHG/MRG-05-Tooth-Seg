"""Torchvision detection backend for Faster R-CNN and RetinaNet."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _torchvision():
    try:
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
        from torchvision.transforms.functional import pil_to_tensor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "torch and torchvision are required for Faster R-CNN/RetinaNet. "
            "Install a matching PyTorch/torchvision pair."
        ) from exc
    return torch, fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, pil_to_tensor


def _paths(data_yaml: Path, split: str) -> tuple[Path, Path]:
    import yaml

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_yaml.parent))
    image_value = config[split]
    image_dir = Path(image_value)
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    label_dir = root / "labels" / split
    return image_dir, label_dir


class YoloDetectionDataset:
    def __init__(self, data_yaml: Path, split: str, pil_to_tensor: Any):
        self.image_dir, self.label_dir = _paths(data_yaml, split)
        self.pil_to_tensor = pil_to_tensor
        self.invalid_boxes = 0
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.images = sorted(p for p in self.image_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
        if not self.images:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        torch = _torchvision()[0]
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        boxes: list[list[float]] = []
        label_path = self.label_dir / f"{path.stem}.txt"
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) != 5:
                    continue
                _, xc, yc, w, h = map(float, values)
                x1, y1 = (xc - w / 2) * width, (yc - h / 2) * height
                x2, y2 = (xc + w / 2) * width, (yc + h / 2) * height
                x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)
                if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                    self.invalid_boxes += 1
                    continue
                boxes.append([x1, y1, x2, y2])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }
        return self.pil_to_tensor(image).float() / 255.0, target, path


def _collate(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)


def build_model(model_name: str, num_classes: int = 2):
    torch, faster, retina, _ = _torchvision()
    key = model_name.lower().replace("-", "").replace("_", "")
    if "faster" in key:
        model = faster(weights=None, weights_backbone=None, num_classes=num_classes)
    elif "retina" in key:
        model = retina(weights=None, weights_backbone=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported torchvision detection model: {model_name}")
    return model


def _device(value: str | None):
    torch = _torchvision()[0]
    requested = (value or "").strip().lower()
    if requested.isdigit():
        requested = f"cuda:{requested}"
    if requested in {"cuda", "cuda:0"} and not torch.cuda.is_available():
        if value is not None:
            raise RuntimeError(
                f"CUDA device '{value}' was requested for the torchvision backend, but CUDA is unavailable. "
                "Use --device cpu or install a CUDA-enabled PyTorch build."
            )
        requested = "cpu"
    return torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))


def train_torchvision_detection(
    data_yaml: Path, model_name: str, epochs: int, imgsz: int, batch: int,
    device: str | None, project: str, name: str, weights: Path | None = None,
) -> Path:
    torch = _torchvision()[0]
    print(f"Torchvision {torch.__version__}  Python-{__import__('sys').version_info.major}.{__import__('sys').version_info.minor} "
          f"device={device or 'auto'}", flush=True)
    print(f"train: Scanning {data_yaml} ...", flush=True)
    dataset = YoloDetectionDataset(data_yaml, "train", _torchvision()[3])
    model = build_model(model_name)
    if weights:
        model.load_state_dict(_checkpoint(weights)["state_dict"])
    dev = _device(device)
    model.to(dev).train()
    loader = torch.utils.data.DataLoader(dataset, batch_size=max(1, batch), shuffle=True, collate_fn=_collate)
    # Detection heads are trained with batch size 1 here; the conservative
    # learning rate and gradient cap prevent a single hard image from making
    # the optimizer state non-finite.
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
    )
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"train: {len(dataset)} images, model={model_name}, parameters={parameter_count:,}", flush=True)
    print(f"Saving to {save_dir.resolve()}", flush=True)
    print("\n      Epoch    GPU_mem   loss       Precision  Recall     mAP50     mAP50-95", flush=True)
    rows = []
    _write_args(save_dir, {
        "model": model_name, "data": str(data_yaml), "epochs": epochs,
        "imgsz": imgsz, "batch": batch, "device": str(dev),
        "backend": "torchvision", "seed": None,
    })
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_index, (images, targets, _) in enumerate(loader, start=1):
            images = [image.to(dev) for image in images]
            targets = [{k: v.to(dev) for k, v in target.items()} for target in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                print(
                    f"WARNING: non-finite loss at epoch {epoch}, batch {batch_index}; "
                    "skipping optimizer step",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            if batch_index == 1 or batch_index == len(loader) or batch_index % max(1, len(loader) // 10) == 0:
                print(f"      {epoch}/{epochs}      batch {batch_index}/{len(loader)}  loss={float(loss.detach().cpu()):.4f}", flush=True)
        metrics = evaluate_torchvision_detection(model, data_yaml, "val", dev)
        if dataset.invalid_boxes:
            print(f"WARNING: skipped {dataset.invalid_boxes} invalid clipped training boxes", flush=True)
        mean_loss = total_loss / max(1, len(loader))
        print(
            f"      {epoch}/{epochs}      {str(dev):<8} {mean_loss:>8.4f} "
            f"{metrics['metrics/precision(B)']:>10.4f} {metrics['metrics/recall(B)']:>8.4f} "
            f"{metrics['metrics/mAP50(B)']:>9.4f} {metrics['metrics/mAP50-95(B)']:>10.4f}",
            flush=True,
        )
        rows.append({"epoch": epoch, "train/loss": mean_loss, **metrics})
        checkpoint = {"model_name": model_name, "state_dict": model.state_dict(), "classes": ["tooth"]}
        torch.save(checkpoint, save_dir / "last.pt")
        if not rows[:-1] or metrics["metrics/mAP50-95(B)"] >= max(r["metrics/mAP50-95(B)"] for r in rows[:-1]):
            torch.save(checkpoint, save_dir / "best.pt")
            print(f"Saved best model to {save_dir / 'best.pt'}", flush=True)
    with (save_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Results saved to {save_dir.resolve()}", flush=True)
    return save_dir / "best.pt"


def _checkpoint(weights: Path) -> dict[str, Any]:
    torch = _torchvision()[0]
    value = torch.load(weights, map_location="cpu", weights_only=False)
    if "state_dict" not in value:
        value = {"state_dict": value, "model_name": "fasterrcnn_resnet50_fpn", "classes": ["tooth"]}
    return value


def load_torchvision_model(weights: Path):
    checkpoint = _checkpoint(weights)
    model = build_model(checkpoint.get("model_name", "fasterrcnn_resnet50_fpn"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def evaluate_torchvision_detection(model, data_yaml: Path, split: str = "val", device=None) -> dict[str, float]:
    torch = _torchvision()[0]
    dev = device or _device(None)
    dataset = YoloDetectionDataset(data_yaml, split, _torchvision()[3])
    print(f"val: Scanning {len(dataset)} images, device={dev}", flush=True)
    model.eval(); tp = fp = fn = 0; ious: list[float] = []
    with torch.no_grad():
        for image, target, _ in (dataset[i] for i in range(len(dataset))):
            output = model([image.to(dev)])[0]
            pred_boxes = output["boxes"].detach().cpu(); scores = output["scores"].detach().cpu()
            keep = scores >= 0.25; pred_boxes = pred_boxes[keep]
            matched: set[int] = set()
            for box in pred_boxes:
                if not len(target["boxes"]): fp += 1; continue
                overlaps = _iou(box, target["boxes"]); value, index = overlaps.max(0)
                if float(value) >= 0.5 and int(index) not in matched:
                    tp += 1; matched.add(int(index)); ious.append(float(value))
                else: fp += 1
            fn += len(target["boxes"]) - len(matched)
    if dataset.invalid_boxes:
        print(f"WARNING: skipped {dataset.invalid_boxes} invalid clipped {split} boxes", flush=True)
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return {
        "metrics/precision(B)": precision,
        "metrics/recall(B)": recall,
        "metrics/mAP50(B)": precision * recall,
        "metrics/mAP50-95(B)": mean_iou * precision * recall,
    }


def _write_args(save_dir: Path, values: dict[str, Any]) -> None:
    try:
        import yaml
        (save_dir / "args.yaml").write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    except ImportError:
        (save_dir / "args.json").write_text(json.dumps(values, indent=2), encoding="utf-8")


def _iou(box, boxes):
    torch = _torchvision()[0]
    lt = torch.maximum(box[:2], boxes[:, :2]); rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0); inter = wh[:, 0] * wh[:, 1]
    area = (box[2] - box[0]) * (box[3] - box[1]); areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter).clamp(min=1e-7)


def validate_torchvision_detection(weights: Path, data_yaml: Path, device: str | None = None) -> dict[str, float]:
    model = load_torchvision_model(weights); return evaluate_torchvision_detection(model, data_yaml, "val", _device(device))


def predict_torchvision(weights: Path, image_path: Path, conf: float = 0.25, device: str | None = None) -> dict[str, Any]:
    torch = _torchvision()[0]; image = Image.open(image_path).convert("RGB"); tensor = _torchvision()[3](image).float() / 255.0
    model = load_torchvision_model(weights).to(_device(device)); model.eval()
    with torch.no_grad(): output = model([tensor.to(_device(device))])[0]
    boxes = output["boxes"].detach().cpu().tolist(); scores = output["scores"].detach().cpu().tolist()
    selected = [(box, score) for box, score in zip(boxes, scores) if score >= conf]
    annotated = image.copy(); draw = ImageDraw.Draw(annotated)
    for box, _ in selected: draw.rectangle(box, outline="#FFFF00", width=5)
    return {"image": annotated, "source": image, "boxes": [x[0] for x in selected], "confs": [x[1] for x in selected], "class_ids": [0] * len(selected), "class_names": ["tooth"] * len(selected), "raw": output}
