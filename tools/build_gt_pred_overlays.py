"""Render ground-truth vs prediction overlay images for the hard subset.

For each candidate image in ``annotation_worksheet.csv`` this draws the
ground-truth boxes (green) and the RT-DETRv2-L prediction boxes (red) on the
original radiograph, so false positives (lone red boxes) and false negatives
(lone green boxes) are visible at a glance. A corner label reports the greedy
IoU match counts (TP / FP / FN) used to define the hard subset.

Inputs:

  <input>/annotation_worksheet.csv                       one row per candidate
  <data>/images/test/<stem>.jpg                          original image
  <data>/labels/test/<stem>.txt                          ground-truth YOLO boxes
  <input>/visualization/<stem>/<stem>_detections.json    prediction boxes (px xyxy)

Output:

  <output>/gt_pred/<stem>_gt_pred.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

IOU_THRESHOLD = 0.5
COLOR_GT = (0, 220, 0)        # green  -> ground truth
COLOR_PRED = (255, 50, 50)    # red    -> prediction
COLOR_TEXT_BG = (0, 0, 0)     # black  -> legend background
COLOR_TEXT = (255, 255, 255)  # white  -> legend text

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _load_gt_boxes(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = map(float, parts)
        x1 = (xc - w / 2.0) * width
        y1 = (yc - h / 2.0) * height
        x2 = (xc + w / 2.0) * width
        y2 = (yc + h / 2.0) * height
        boxes.append([x1, y1, x2, y2])
    return boxes


def _load_pred_boxes(det_json: Path) -> list[tuple[list[float], float]]:
    if not det_json.exists():
        return []
    data = json.loads(det_json.read_text(encoding="utf-8"))
    preds = []
    for det in data.get("detections", []):
        box = det["box_xyxy"]
        preds.append(([float(v) for v in box], float(det.get("confidence", 0.0))))
    preds.sort(key=lambda item: item[1], reverse=True)
    return preds


def _match(preds: list[tuple[list[float], float]], gts: list[list[float]]) -> tuple[int, int, int]:
    matched_gt: set[int] = set()
    tp = 0
    for pbox, _ in preds:
        best_iou, best_gt = 0.0, -1
        for g_idx, gbox in enumerate(gts):
            if g_idx in matched_gt:
                continue
            value = _iou(pbox, gbox)
            if value > best_iou:
                best_iou, best_gt = value, g_idx
        if best_gt >= 0 and best_iou >= IOU_THRESHOLD:
            matched_gt.add(best_gt)
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def _draw_legend(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, pad: int = 6) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.rectangle((0, 0, tw + pad * 2, th + pad * 2), fill=COLOR_TEXT_BG)
    draw.text((pad, pad), text, fill=COLOR_TEXT, font=font)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/hard-subset")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/OMNI_COCO/after")
    parser.add_argument("--output", type=Path, default=ROOT / "output/hard-subset/gt_pred")
    args = parser.parse_args()

    csv_path = args.input / "annotation_worksheet.csv"
    vis_dir = args.input / "visualization"
    image_dir = args.data / "images" / "test"
    label_dir = args.data / "labels" / "test"

    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Index source images by stem (case-insensitive).
    image_index = {
        p.stem.lower(): p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }

    args.output.mkdir(parents=True, exist_ok=True)
    font = _font(16)

    missing: list[str] = []
    written = 0
    for row in rows:
        stem = row["图像"]
        src = image_index.get(stem.lower())
        if src is None:
            missing.append(stem)
            continue

        image = Image.open(src).convert("RGB")
        width, height = image.size
        gts = _load_gt_boxes(label_dir / f"{stem}.txt", width, height)
        preds = _load_pred_boxes(vis_dir / stem / f"{stem}_detections.json")
        tp, fp, fn = _match(preds, gts)

        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for box in gts:
            draw.rectangle(box, outline=COLOR_GT, width=4)
        for box, _ in preds:
            draw.rectangle(box, outline=COLOR_PRED, width=3)

        text = f"{stem}  GT={len(gts)} pred={len(preds)} TP={tp} FP={fp} FN={fn}"
        _draw_legend(draw, text, font)

        out_path = args.output / f"{stem}_gt_pred.png"
        canvas.save(out_path)
        written += 1

    print(f"Wrote {written} overlays to {args.output}")
    if missing:
        print(f"WARNING: {len(missing)} source images missing: {missing[:20]}")


if __name__ == "__main__":
    main()
