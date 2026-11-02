"""Run fixed test-set validation for every comparison model and aggregate the metrics.

Uses the manually reviewed ``after`` labels (``dataset/OMNI_COCO/after``) as ground
truth and evaluates on the fixed ``test`` split. Requires the training environment
(ultralytics / torch / torchvision / transformers) to be installed.

Notes
-----
* RF-DETR-M is excluded: its native ``validate`` reads its own materialized ``valid``
  view and does not accept a ``--split`` argument, so a test-set result cannot be
  produced through the same protocol.
* RT-DETRv2-L reports mAP through ``_evaluate_rtdetrv2`` (same metric keys as the
  torchvision adapter).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toothseg.data import materialize_yolo_data_yaml  # noqa: E402
from toothseg.model_catalog import require_trainable_model  # noqa: E402

METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)

# (name, weights path relative to ROOT, model_name, split_supported)
MODELS = [
    ("yolo11n", "output/train/omni_coco/yolo11n/omni-coco-after-result/toothseg-output/train/omni-coco-after-v4/weights/best.pt", "yolo11n.pt", True),
    ("yolo11s", "output/train/omni_coco/yolo11s/omni-coco-after-result/toothseg-output/train/omni-coco-after-v4/weights/best.pt", "yolo11s.pt", True),
    ("yolov8m", "output/train/omni_coco/yolov8m/omni-coco-yolov8m-after-result/toothseg-output/train/omni-coco-yolov8m-after/weights/best.pt", "yolov8m.pt", True),
    ("yolo26m", "output/train/omni_coco/yolo26m/omni-coco-yolo26m-after-result/toothseg-output/train/omni-coco-yolo26m-after/weights/best.pt", "yolo26m.pt", True),
    ("rtdetr-l", "output/train/omni_coco/rtdetr-l/omni-coco-rtdetr-l-after-v4-result/toothseg-output/train/omni-coco-rtdetr-l-after-v4/weights/best.pt", "rtdetr-l.pt", True),
    ("faster-r-cnn", "output/train/omni_coco/faster-r-cnn/omni-coco-after-fasterrcnn/best.pt", "fasterrcnn_resnet50_fpn", True),
    ("retinanet", "output/train/omni_coco/retinanet/omni-coco-after-retinanet/best.pt", "retinanet_resnet50_fpn", True),
    ("rtdetrv2-l", "output/train/omni_coco/rtdetrv2-l/omni-coco-after-rtdetrv2l", "rtdetrv2-l", True),
    ("rf-detr-m", "output/train/omni_coco/rfdetr-m/omni-coco-after-rfdetrm/checkpoint_best_total.pth", "rf-detr-m", False),
]


def evaluate_model(model_name: str, weights: Path, data_yaml: Path, split: str) -> dict:
    spec = require_trainable_model(model_name)
    if spec.backend == "torchvision":
        from toothseg.torchvision_adapter import validate_torchvision_detection
        return validate_torchvision_detection(weights, data_yaml, split=split)
    if spec.backend in {"transformers", "rfdetr"}:
        from toothseg.external_adapter import validate_external_detection
        return validate_external_detection(weights, data_yaml, spec.name, split=split)
    from ultralytics import YOLO
    result = YOLO(str(weights)).val(data=str(data_yaml), split=split, verbose=False)
    metrics = getattr(result, "results_dict", None)
    if metrics is None:
        metrics = result if isinstance(result, dict) else {}
    return metrics


def _format(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/OMNI_COCO/after")
    parser.add_argument("--output", type=Path, default=ROOT / "output/val-test")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model names to run.")
    args = parser.parse_args()

    data_yaml = materialize_yolo_data_yaml(args.data, args.output / ".runtime")
    selected = [m for m in MODELS if args.models is None or m[0] in args.models]
    if args.models is not None:
        known = {m[0] for m in MODELS}
        for name in args.models:
            if name not in known:
                print(f"WARNING: unknown model '{name}' (known: {', '.join(sorted(known))})")

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name, rel_weights, model_name, split_supported in selected:
        weights = ROOT / rel_weights
        row = {"model": name}
        if not split_supported:
            print(f"SKIP  {name}: native validate does not support --split (uses its own valid view)")
            row["status"] = "skipped (no split support)"
            rows.append(row)
            continue
        if not weights.exists():
            print(f"SKIP  {name}: weights not found: {weights}")
            row["status"] = "missing weights"
            rows.append(row)
            continue
        print(f"EVAL  {name}  split={args.split}  weights={rel_weights}")
        try:
            metrics = evaluate_model(model_name, weights, data_yaml, args.split)
            row["status"] = "ok"
            for key in METRIC_KEYS:
                row[key.rsplit("/", 1)[-1]] = metrics.get(key)
            print("      " + "  ".join(f"{key.rsplit('/', 1)[-1]}={_format(metrics.get(key))}" for key in METRIC_KEYS))
        except Exception as exc:  # noqa: BLE001 - keep the batch going across backends
            print(f"FAIL  {name}: {exc}")
            row["status"] = f"error: {exc}"
        rows.append(row)

    fieldnames = ["model", "status", *[key.rsplit("/", 1)[-1] for key in METRIC_KEYS]]
    csv_path = args.output / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# test 集统一评估结果",
        "",
        f"- 数据：`{args.data}`（after 人工复核标签作为 ground truth）",
        f"- split：`{args.split}`",
        f"- 指标来源：各后端验证返回的 `metrics/precision(B)`、`metrics/recall(B)`、`metrics/mAP50(B)`、`metrics/mAP50-95(B)`",
        "- RF-DETR-M 原生 `validate` 读取自身物化的 `valid` 视图，不接受 `--split`，因此未纳入本表。",
        "",
        "| 模型 | P(B) | R(B) | mAP50(B) | mAP50-95(B) | 状态 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {_format(row.get('precision(B)'))} | {_format(row.get('recall(B)'))} "
            f"| {_format(row.get('mAP50(B)'))} | {_format(row.get('mAP50-95(B)'))} | {row['status']} |"
        )
    lines.append("")
    lines.append(f"原始 CSV：`{csv_path.name}`")
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {csv_path} and README.md to {args.output}")


if __name__ == "__main__":
    main()
