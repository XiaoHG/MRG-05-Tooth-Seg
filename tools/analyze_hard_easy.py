"""Hard-vs-easy before/after analysis on the full fixed test set.

The hard set is the manually-selected ``hard_diff`` (32 stems); every other
test image is the easy set. For each image we recompute, against the *after*
ground-truth labels, the before-model and after-model TP/FP/FN (greedy
IoU >= 0.5), then aggregate the two groups separately. This answers whether
training-label review helps more on hard samples than on easy ones.

Inputs:

  <base>/hard_diff/*.png                     hard stems
  <base>/before/predictions.json             before-model detections (828)
  <base>/after/predictions.json              after-model detections (828)
  <data>/labels/test/<stem>.txt              after ground-truth (YOLO)

Outputs:

  <base>/hard_easy_analysis.csv              per-image rows (group, before/after)
  <base>/hard_easy_analysis.md               group-level summary + deltas
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


def _detections(predictions: dict, stem: str) -> tuple[list[tuple[list[float], float]], int, int]:
    for key, value in predictions.items():
        if Path(key).stem.lower() == stem.lower():
            width = int(value.get("image_width", 512))
            height = int(value.get("image_height", 341))
            dets = [
                (det["box_xyxy"], float(det.get("confidence", 0.0)))
                for det in value.get("detections", [])
            ]
            dets.sort(key=lambda item: item[1], reverse=True)
            return dets, width, height
    return [], 512, 341


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
    return tp, len(preds) - tp, len(gts) - tp


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

    hard_stems = {p.stem for p in hard_dir.glob("*.png")}
    if not hard_stems:
        raise SystemExit(f"No PNGs in {hard_dir}")

    before = json.loads((base / "before" / "predictions.json").read_text(encoding="utf-8"))
    after = json.loads((base / "after" / "predictions.json").read_text(encoding="utf-8"))

    # All test stems (before/after keys should coincide with the 828 test images).
    all_stems = sorted({Path(k).stem for k in after.keys()} | {Path(k).stem for k in before.keys()})
    missing_hard = sorted(hard_stems - set(all_stems))
    if missing_hard:
        print(f"WARNING: {len(missing_hard)} hard stems not in predictions: {missing_hard[:10]}")

    # Accumulators: group -> model -> dict of tp/fp/fn/gt/err
    stats = {"hard": {"before": {}, "after": {}}, "easy": {"before": {}, "after": {}}}
    rows = []
    for stem in all_stems:
        group = "hard" if stem in hard_stems else "easy"
        det_b, w_b, h_b = _detections(before, stem)
        det_a, w_a, h_a = _detections(after, stem)
        gts = _load_gt(label_dir / f"{stem}.txt", w_a, h_a)

        tp_b, fp_b, fn_b = _match(det_b, gts)
        tp_a, fp_a, fn_a = _match(det_a, gts)

        for key, (tp, fp, fn) in (("before", (tp_b, fp_b, fn_b)), ("after", (tp_a, fp_a, fn_a))):
            s = stats[group][key]
            s["gt"] = s.get("gt", 0) + len(gts)
            s["tp"] = s.get("tp", 0) + tp
            s["fp"] = s.get("fp", 0) + fp
            s["fn"] = s.get("fn", 0) + fn

        rows.append({
            "image": stem, "group": group, "gt": len(gts),
            "b_tp": tp_b, "b_fp": fp_b, "b_fn": fn_b,
            "a_tp": tp_a, "a_fp": fp_a, "a_fn": fn_a,
        })

    # CSV.
    csv_path = base / "hard_easy_analysis.csv"
    fieldnames = ["image", "group", "gt", "b_tp", "b_fp", "b_fn", "a_tp", "a_fp", "a_fn"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary table.
    def metrics(s):
        return _ratio(s["tp"], s["tp"] + s["fp"]), _ratio(s["tp"], s["tp"] + s["fn"]), s["fp"] + s["fn"]

    lines = [
        "# 困难 / 易样本 before/after 对比",
        "",
        f"- 困难样本：`hard_diff/` 人工挑选 **{len(hard_stems)}** 张；易样本：其余 **{len(all_stems) - len(hard_stems)}** 张",
        f"- 真值固定为 `after` 人工复核标签；匹配 IoU ≥ {IOU_THRESHOLD}",
        "",
        "| 组 | 样本 | GT | 模型 | TP | FP | FN | P | R | 错误 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group, label in (("hard", "困难"), ("easy", "易")):
        s = stats[group]
        pb, rb, eb = metrics(s["before"])
        pa, ra, ea = metrics(s["after"])
        n = len(hard_stems) if group == "hard" else len(all_stems) - len(hard_stems)
        lines.append(f"| {label} | {n} | {s['before']['gt']} | before | {s['before']['tp']} | {s['before']['fp']} | {s['before']['fn']} | {pb:.3f} | {rb:.3f} | {eb} |")
        lines.append(f"| {label} | {n} | {s['after']['gt']} | after | {s['after']['tp']} | {s['after']['fp']} | {s['after']['fn']} | {pa:.3f} | {ra:.3f} | {ea} |")
    lines.append("")
    lines.append("## 复核收益（after − before，负数=改善）")
    lines.append("")
    lines.append("| 组 | ΔFP | ΔFN | Δ错误 | 错误相对降幅 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for group, label in (("hard", "困难"), ("easy", "易")):
        s = stats[group]
        d_fp = s["after"]["fp"] - s["before"]["fp"]
        d_fn = s["after"]["fn"] - s["before"]["fn"]
        e_b = s["before"]["fp"] + s["before"]["fn"]
        e_a = s["after"]["fp"] + s["after"]["fn"]
        rel = (e_a - e_b) / e_b if e_b else 0.0
        lines.append(f"| {label} | {d_fp:+d} | {d_fn:+d} | {e_a - e_b:+d} | {rel:+.1%} |")
    lines.append("")
    lines.append(f"原始数据：`{csv_path.name}`")

    md_path = base / "hard_easy_analysis.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"hard={len(hard_stems)} easy={len(all_stems)-len(hard_stems)} total={len(all_stems)}")
    for group, label in (("hard", "困难"), ("easy", "易")):
        s = stats[group]
        pb, rb, eb = metrics(s["before"]); pa, ra, ea = metrics(s["after"])
        print(f"{label}: before FP={s['before']['fp']} FN={s['before']['fn']} err={eb} P={pb:.3f} R={rb:.3f}  "
              f"| after FP={s['after']['fp']} FN={s['after']['fn']} err={ea} P={pa:.3f} R={ra:.3f}")
    print(f"Wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
