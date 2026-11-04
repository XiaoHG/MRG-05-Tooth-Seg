"""Consolidated fixed-test evaluation: before vs after weights under the SAME GT.

The per-variant runs in ``eval_test_set.py`` pair each model's weights with the
matching label version (before weights on before labels, after weights on after
labels). That diagonal cannot separate "training-label review effect" from
"ground-truth difference". This script computes the full 2x2 (weights variant x
ground-truth variant) so the review effect is measured holding the ground truth
fixed.

Main output is the fixed **after** ground-truth table: for each of the 8 models,
the before-trained and after-trained weights are both evaluated against the after
labels; the delta is the training-label review effect on detection quality. A
symmetric fixed-before table and within-GT rankings are also written.

Inputs (reused from ``eval_test_set.py``):
  MODELS / evaluate_model / METRIC_KEYS

Outputs:
  <output>/results.csv                 one row per (model, weights, gt)
  <output>/README.md                   fixed-after + fixed-before tables + rankings
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for _p in (SRC, TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_test_set import MODELS, METRIC_KEYS, evaluate_model  # noqa: E402
from toothseg.data import materialize_yolo_data_yaml  # noqa: E402

VARIANTS = ["before", "after"]
KEY_LABELS = {
    "precision(B)": "P",
    "recall(B)": "R",
    "mAP50(B)": "mAP50",
    "mAP50-95(B)": "mAP50-95",
}


def _float(value):
    return float(value) if isinstance(value, (int, float)) else None


def _fmt(value):
    return f"{value:.4f}" if value is not None else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/val-test-cross")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    selected = [m for m in MODELS if args.models is None or m[0] in args.models]
    args.output.mkdir(parents=True, exist_ok=True)

    # materialize the two data views once
    data_yamls = {
        v: materialize_yolo_data_yaml(ROOT / "dataset/OMNI_COCO" / v, args.output / ".runtime" / v)
        for v in VARIANTS
    }

    rows: list[dict] = []
    for name, weights_by_variant, model_name in selected:
        for wvar in VARIANTS:
            weights = ROOT / weights_by_variant[wvar]
            if not weights.exists():
                print(f"SKIP  {name}/{wvar}: weights not found: {weights}")
                continue
            for gvar in VARIANTS:
                print(f"EVAL  {name}  weights={wvar}  gt={gvar}  split={args.split}")
                try:
                    metrics = evaluate_model(model_name, weights, data_yamls[gvar], args.split)
                    row = {"model": name, "weights": wvar, "gt": gvar, "status": "ok"}
                    for key in METRIC_KEYS:
                        short = key.rsplit("/", 1)[-1]
                        row[short] = _float(metrics.get(key))
                    print("      " + "  ".join(f"{KEY_LABELS[k]}={_fmt(row[k])}" for k in KEY_LABELS))
                except Exception as exc:  # noqa: BLE001 - keep the batch going across backends
                    print(f"FAIL  {name}/{wvar}/{gvar}: {exc}")
                    row = {"model": name, "weights": wvar, "gt": gvar, "status": f"error: {exc}"}
                rows.append(row)

    # CSV.
    fieldnames = ["model", "weights", "gt", "status", *KEY_LABELS.keys()]
    csv_path = args.output / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Index rows by (model, weights, gt).
    idx = {(r["model"], r["weights"], r["gt"]): r for r in rows}

    def table(gt_fixed: str, title: str) -> list[str]:
        lines = [
            f"## 固定 {gt_fixed} 真值（{title}）",
            "",
            "| 模型 | before P | before R | before mAP50 | before mAP50-95 | after P | after R | after mAP50 | after mAP50-95 | ΔmAP50-95 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, _, _ in selected:
            b = idx.get((name, "before", gt_fixed))
            a = idx.get((name, "after", gt_fixed))
            if not b or not a:
                lines.append(f"| {name} | 缺失 | | | | | | | | |")
                continue
            delta = (a["mAP50-95(B)"] or 0) - (b["mAP50-95(B)"] or 0)
            lines.append(
                f"| {name} | {_fmt(b['precision(B)'])} | {_fmt(b['recall(B)'])} | {_fmt(b['mAP50(B)'])} | {_fmt(b['mAP50-95(B)'])} "
                f"| {_fmt(a['precision(B)'])} | {_fmt(a['recall(B)'])} | {_fmt(a['mAP50(B)'])} | {_fmt(a['mAP50-95(B)'])} | {delta:+.4f} |"
            )
        return lines

    def ranking(gt_fixed: str, weights: str, title: str) -> list[str]:
        scored = []
        for name, _, _ in selected:
            r = idx.get((name, weights, gt_fixed))
            if r and r["mAP50-95(B)"] is not None:
                scored.append((r["mAP50-95(B)"], name))
        scored.sort(key=lambda item: item[0], reverse=True)
        lines = [f"| 排名 | 模型 | mAP50-95 |", "| --- | --- | --- |"]
        for i, (v, name) in enumerate(scored, 1):
            lines.append(f"| {i} | {name} | {_fmt(v)} |")
        return [f"### {title}（按 mAP50-95）", ""] + lines

    lines = [
        "# test 集 2×2 交叉评估（固定真值下的 before/after 对比）",
        "",
        f"- split：`{args.split}`",
        f"- 指标：P/R/mAP50/mAP50-95，来源各后端验证返回的 `metrics/*(B)`",
        "- `before` / `after` 指训练所用的标签版本；`gt` 指评估所用的真值版本。",
        "",
    ]
    lines += table("after", "复核收益，真值固定为人工复核后标签")
    lines += [""]
    lines += ranking("after", "after", "固定 after 真值 · after 权重排名")
    lines += [""]
    lines += ranking("after", "before", "固定 after 真值 · before 权重排名")
    lines += [""]
    lines += table("before", "参考，真值固定为复核前标签")
    lines += [""]
    lines.append(f"原始 CSV：`{csv_path.name}`")

    md_path = args.output / "README.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
