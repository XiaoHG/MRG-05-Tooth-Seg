"""Screen the fixed test set for hard samples: per-image TP/FP/FN + image features.

Purpose (v13 task 2): produce an *automatic shortlist* of images where a strong model
mis-detects (FP) or misses (FN) teeth, ranked by error count, with brightness and
highlight-ratio features. This is NOT the hard-subset itself — the semantic labels
(glare / metal / occlusion / incisor) still need a human to confirm on the shortlist.

The model is loaded ONCE and reused across images (each backend's ``predict_*`` helper
reloads weights per call, which is far too slow for a full test split).

The script matches predicted boxes against ground-truth YOLO boxes with a greedy
IoU match (predictions processed by descending confidence, IoU threshold 0.5 by
default), then writes:

  <output>/screen.csv      one row per image, sorted by (fp + fn) descending
  <output>/errors.json     per-image FP/FN box details (normalized xywh + confidence)
  <output>/README.md       a top-N markdown summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toothseg.model import _read_checkpoint_metadata  # noqa: E402
from toothseg.model_catalog import find_model_spec  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HIGHLIGHT_GRAY = 245.0  # pixels at/above this gray level count as "highlight"


class Detector:
    """Load a detection model once and reuse it across images."""

    def __init__(self, weights: Path, conf: float, device: str | None):
        metadata = _read_checkpoint_metadata(weights)
        model_name = metadata.get("model_name", weights.name) if isinstance(metadata, dict) else weights.name
        spec = find_model_spec(model_name)
        self.backend = spec.backend if spec is not None else "ultralytics"
        self.conf = conf
        self.device = device
        self._load(weights)

    def _load(self, weights: Path) -> None:
        if self.backend == "torchvision":
            import torch
            from torchvision.transforms.functional import pil_to_tensor

            from toothseg.torchvision_adapter import load_torchvision_model

            self._torch = torch
            self._pil_to_tensor = pil_to_tensor
            requested = (self.device or "").strip().lower()
            if requested.isdigit():
                requested = f"cuda:{requested}"
            if requested in {"cuda", "cuda:0"} and not torch.cuda.is_available():
                requested = "cpu"
            self._device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
            self._model = load_torchvision_model(weights).to(self._device).eval()
        elif self.backend == "transformers":
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            self._torch = torch
            self._processor = AutoImageProcessor.from_pretrained(str(weights))
            self._model = AutoModelForObjectDetection.from_pretrained(str(weights))
            use_cuda = torch.cuda.is_available() and (self.device is None or "cpu" not in str(self.device).lower())
            self._device = torch.device("cuda" if use_cuda else "cpu")
            self._model.to(self._device).eval()
        else:  # ultralytics
            from ultralytics import YOLO

            self._model = YOLO(str(weights))

    def predict(self, image_path: Path) -> tuple[list[list[float]], list[float]]:
        """Return (boxes_xyxy, confidences) for one image."""
        if self.backend == "ultralytics":
            kwargs: dict[str, Any] = {"source": str(image_path), "conf": self.conf, "verbose": False}
            if self.device:
                kwargs["device"] = self.device
            result = self._model.predict(**kwargs)[0]
            if getattr(result, "boxes", None) is None:
                return [], []
            return result.boxes.xyxy.tolist(), [float(x) for x in result.boxes.conf.tolist()]
        if self.backend == "torchvision":
            image = Image.open(image_path).convert("RGB")
            tensor = self._pil_to_tensor(image).float() / 255.0
            with self._torch.no_grad():
                output = self._model([tensor.to(self._device)])[0]
            boxes = output["boxes"].detach().cpu().tolist()
            scores = output["scores"].detach().cpu().tolist()
            selected_boxes, selected_scores = [], []
            for box, score in zip(boxes, scores):
                if float(score) >= self.conf:
                    selected_boxes.append(box)
                    selected_scores.append(float(score))
            return selected_boxes, selected_scores
        # transformers
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        prediction = self._processor.post_process_object_detection(
            outputs, threshold=self.conf, target_sizes=[(image.height, image.width)]
        )[0]
        return prediction["boxes"].tolist(), [float(x) for x in prediction["scores"].tolist()]


def load_yolo_boxes(label_path: Path) -> list[list[float]]:
    """Return normalized [x_center, y_center, width, height] boxes from a YOLO label file."""
    boxes: list[list[float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return boxes


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0]


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def match_boxes(
    preds: list[tuple[list[float], float]], gts: list[list[float]], iou_thr: float
) -> tuple[int, int, int, list[dict], list[dict]]:
    """Greedy IoU match. Returns (tp, fp, fn, fp_details, fn_details)."""
    matched_gt: set[int] = set()
    tp = 0
    fp_details: list[dict] = []
    for pbox, conf in preds:  # preds already sorted by descending confidence
        best_iou, best_gt = 0.0, -1
        for g_idx, gbox in enumerate(gts):
            if g_idx in matched_gt:
                continue
            value = _iou(pbox, gbox)
            if value > best_iou:
                best_iou, best_gt = value, g_idx
        if best_gt >= 0 and best_iou >= iou_thr:
            matched_gt.add(best_gt)
            tp += 1
        else:
            fp_details.append({"box": pbox, "conf": conf, "best_iou": best_iou})

    fn_details = [
        {"box": gbox, "conf": None, "best_iou": None}
        for g_idx, gbox in enumerate(gts)
        if g_idx not in matched_gt
    ]
    return tp, len(fp_details), len(fn_details), fp_details, fn_details


def image_features(image_path: Path) -> tuple[float, float]:
    gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    highlight_ratio = float((gray >= HIGHLIGHT_GRAY).mean())
    return brightness, highlight_ratio


def _normalized_xywh(xyxy: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = xyxy
    return [
        (x1 + x2) / 2.0 / width,
        (y1 + y2) / 2.0 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Model weights (best.pt or backend dir).")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/OMNI_COCO/after")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "output/hard-subset")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top", type=int, default=50, help="Rows in the markdown summary.")
    args = parser.parse_args()

    image_dir = args.data / "images" / args.split
    label_dir = args.data / "labels" / args.split
    if not image_dir.is_dir():
        raise SystemExit(f"Image dir not found: {image_dir}")

    image_paths = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise SystemExit(f"No images found in: {image_dir}")

    args.output.mkdir(parents=True, exist_ok=True)
    detector = Detector(args.weights, args.conf, args.device)
    print(f"Loaded detector (backend={detector.backend}). Scanning {len(image_paths)} images...")

    rows: list[dict] = []
    errors: dict[str, dict] = {}
    total_tp = total_fp = total_fn = 0
    error_image_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        stem = image_path.stem
        label_path = label_dir / f"{stem}.txt"
        gts = load_yolo_boxes(label_path)

        boxes, confs = detector.predict(image_path)
        width, height = Image.open(image_path).size
        preds = [(_normalized_xywh(box, width, height), conf) for box, conf in zip(boxes, confs)]
        preds.sort(key=lambda item: item[1], reverse=True)

        tp, fp, fn, fp_details, fn_details = match_boxes(preds, gts, args.iou)
        brightness, highlight_ratio = image_features(image_path)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        if fp or fn:
            error_image_count += 1

        rows.append(
            {
                "image": stem,
                "gt_count": len(gts),
                "pred_count": len(preds),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "error": fp + fn,
                "brightness": round(brightness, 2),
                "highlight_ratio": round(highlight_ratio, 4),
            }
        )
        if fp or fn:
            errors[stem] = {"fp": fp_details, "fn": fn_details}

        if index % 50 == 0 or index == len(image_paths):
            print(f"[{index}/{len(image_paths)}] {stem}  tp={tp} fp={fp} fn={fn}")

    rows.sort(key=lambda r: (-r["error"], -r["fp"]))

    csv_path = args.output / "screen.csv"
    fieldnames = [
        "image", "gt_count", "pred_count", "tp", "fp", "fn", "error",
        "brightness", "highlight_ratio",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output / "errors.json"
    json_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    top_rows = [r for r in rows if r["error"] > 0][: args.top]
    lines = [
        "# 困难子集初筛结果",
        "",
        f"- 数据：`{args.data}`（`{args.split}` split，{len(image_paths)} 张）",
        f"- 权重：`{args.weights}`（backend={detector.backend}）",
        f"- 匹配：IoU ≥ {args.iou}，置信度阈值 {args.conf}",
        "",
        f"- 总 TP = {total_tp}，总 FP = {total_fp}，总 FN = {total_fn}",
        f"- 有错误（FP+FN > 0）的图像：{error_image_count} / {len(image_paths)}",
        "",
        "> 说明：本表仅为**自动初筛候选**。困难子集的语义标签（高光/金属/遮挡/门牙）需人工在这些候选图上确认。",
        "",
        f"## 错误最多的前 {len(top_rows)} 张候选图",
        "",
        "| 图像 | GT | 预测 | TP | FP | FN | 错误 | 亮度 | 高光占比 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in top_rows:
        lines.append(
            f"| {r['image']} | {r['gt_count']} | {r['pred_count']} | {r['tp']} | {r['fp']} | {r['fn']} "
            f"| {r['error']} | {r['brightness']} | {r['highlight_ratio']} |"
        )
    lines.append("")
    lines.append(f"完整清单：`{csv_path.name}`；FP/FN 框明细：`{json_path.name}`")
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote {csv_path}, {json_path}, and README.md to {args.output}")
    print(f"Total TP={total_tp} FP={total_fp} FN={total_fn}; images with errors={error_image_count}/{len(image_paths)}")


if __name__ == "__main__":
    main()
