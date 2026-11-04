"""Analyse the manually-selected hard samples as a before/after comparison.

The hard samples are the PNG stems present in ``hard_diff`` (each a side-by-side
before/after composite chosen by the reviewer). For each stem this recomputes,
against the *after* ground-truth labels, the before-model and after-model
TP/FP/FN (greedy IoU >= 0.5) and then compares them. This isolates the effect of
training-label review (before vs after) on the hard subset, with the ground truth
held fixed.

Inputs:

  <base>/hard_diff/*.png                              hard-sample stems
  <base>/before/predictions.json                      before-model detections
  <base>/after/predictions.json                       after-model detections
  <data>/labels/test/<stem>.txt                       after ground-truth (YOLO)

Outputs:

  <base>/hard_diff_analysis.csv                       per-sample before/after rows
  <base>/hard_diff_analysis.md                        summary + per-sample table
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IOU_THRESHOLD = 0.5


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


def _load_gt(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = map(float, parts)
        boxes.append([
            (xc - w / 2.0) * width, (yc - h / 2.0) * height,
            (xc + w / 2.0) * width, (yc + h / 2.0) * height,
        ])
    return boxes


def _detections(predictions: dict, stem: str, width: int, height: int) -> list[tuple[list[float], float]]:
    # predictions keyed by "<stem>.JPG"; match case-insensitively by stem.
    for key, value in predictions.items():
        if Path(key).stem.lower() == stem.lower():
            width = value.get("image_width", width)
            height = value.get("image_height", height)
            dets = [
                (det["box_xyxy"], float(det.get("confidence", 0.0)))
                for det in value.get("detections", [])
            ]
            dets.sort(key=lambda item: item[1], reverse=True)
            return dets, width, height
    return [], width, height


def _match(preds: list[tuple[list[float], float]], gts: list[list[float]]) -> tuple[int, int, int]:
    matched: set[int] = set()
    tp = 0
    for pbox, _ in preds:
        best_iou, best = 0.0, -1
        for g_idx, gbox in enumerate(gts):
            if g_idx in matched:
                continue
            value = _iou(pbox, gbox)
            if value > best_iou:
                best_iou, best = value, g_idx
        if best >= 0 and best_iou >= IOU_THRESHOLD:
            matched.add(best)
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def _ratio(tp: int, denom: int) -> float:
    return tp / denom if denom > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "output/hard-subset/visualization")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/OMNI_COCO/after")
    args = parser.parse_args()

    base = args.base
    hard_dir = base / "hard_diff"
    label_dir = args.data / "labels" / "test"

    stems = sorted(p.stem for p in hard_dir.glob("*.png"))
    if not stems:
        raise SystemExit(f"No PNGs in {hard_dir}")

    before = json.loads((base / "before" / "predictions.json").read_text(encoding="utf-8"))
    after = json.loads((base / "after" / "predictions.json").read_text(encoding="utf-8"))

    rows = []
    totals = {"gt": 0, "before_tp": 0, "before_fp": 0, "before_fn": 0, "after_tp": 0, "after_fp": 0, "after_fn": 0}
    for stem in stems:
        # Use the first detection to recover image dimensions; fall back to label-derived
        # dims via the predictions' own image_width/height.
        det_b, w, h = _detections(before, stem, 512, 341)
        det_a, w, h = _detections(after, stem, w, h)
        gts = _load_gt(label_dir / f"{stem}.txt", w, h)

        tp_b, fp_b, fn_b = _match(det_b, gts)
        tp_a, fp_a, fn_a = _match(det_a, gts)

        totals["gt"] += len(gts)
        totals["before_tp"] += tp_b
        totals["before_fp"] += fp_b
        totals["before_fn"] += fn_b
        totals["after_tp"] += tp_a
        totals["after_fp"] += fp_a
        totals["after_fn"] += fn_a

        rows.append({
            "image": stem, "gt": len(gts),
            "b_pred": len(det_b), "b_tp": tp_b, "b_fp": fp_b, "b_fn": fn_b,
            "a_pred": len(det_a), "a_tp": tp_a, "a_fp": fp_a, "a_fn": fn_a,
            "d_fp": fp_a - fp_b, "d_fn": fn_a - fn_b,
        })

    # Summary metrics.
    def summary(tp, fp, fn):
        return {
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "error": fp + fn,
        }
    b = summary(totals["before_tp"], totals["before_fp"], totals["before_fn"])
    a = summary(totals["after_tp"], totals["after_fp"], totals["after_fn"])

    # CSV.
    csv_path = base / "hard_diff_analysis.csv"
    fieldnames = ["image", "gt", "b_pred", "b_tp", "b_fp", "b_fn", "a_pred", "a_tp", "a_fp", "a_fn", "d_fp", "d_fn"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Markdown.
    lines = [
        "# 困难样本 before/after 分析",
        "",
        f"- 困难样本数：**{len(stems)}**（来自 `hard_diff/`，人工挑选）",
        f"- 真值：`after` 人工复核标签（固定不变）；匹配 IoU ≥ {IOU_THRESHOLD}",
        "",
        "## 汇总（before 模型 vs after 模型，同一批 32 张困难样本）",
        "",
        "| 指标 | before | after | Δ（after−before） |",
        "| --- | --- | --- | --- |",
        f"| GT 牙数 | {totals['gt']} | {totals['gt']} | — |",
        f"| TP | {totals['before_tp']} | {totals['after_tp']} | {totals['after_tp'] - totals['before_tp']:+d} |",
        f"| FP（误检） | {totals['before_fp']} | {totals['after_fp']} | {totals['after_fp'] - totals['before_fp']:+d} |",
        f"| FN（漏检） | {totals['before_fn']} | {totals['after_fn']} | {totals['after_fn'] - totals['before_fn']:+d} |",
        f"| 精度 P | {b['precision']:.3f} | {a['precision']:.3f} | {a['precision'] - b['precision']:+.3f} |",
        f"| 召回 R | {b['recall']:.3f} | {a['recall']:.3f} | {a['recall'] - b['recall']:+.3f} |",
        f"| 总错误 FP+FN | {b['error']} | {a['error']} | {a['error'] - b['error']:+d} |",
        "",
        "## 逐样本",
        "",
        "| 图像 | GT | before pred/FP/FN | after pred/FP/FN | ΔFP | ΔFN |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['image']} | {r['gt']} | {r['b_pred']}/{r['b_fp']}/{r['b_fn']} "
            f"| {r['a_pred']}/{r['a_fp']}/{r['a_fn']} | {r['d_fp']:+d} | {r['d_fn']:+d} |"
        )
    lines.append("")
    lines.append(f"原始数据：`{csv_path.name}`")

    md_path = base / "hard_diff_analysis.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Hard samples: {len(stems)}")
    print(f"before  TP={totals['before_tp']} FP={totals['before_fp']} FN={totals['before_fn']} P={b['precision']:.3f} R={b['recall']:.3f}")
    print(f"after   TP={totals['after_tp']} FP={totals['after_fp']} FN={totals['after_fn']} P={a['precision']:.3f} R={a['recall']:.3f}")
    print(f"dFP={totals['after_fp']-totals['before_fp']:+d}  dFN={totals['after_fn']-totals['before_fn']:+d}")
    print(f"Wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
