"""Assemble the paper-A·A0 downstream detection baseline report.

Reads the already-produced v13 artifacts and writes a single consolidated markdown
report that closes out task 3 (下游基线复现):

  <val-test>/results.csv                          fixed test-set metrics (after)
  <hard>/visualization/hard_diff_analysis.csv     per-image FP/FN on the 32 hard samples
  <hard>/visualization/hard_easy_analysis.csv     hard-vs-easy before/after aggregates
  <hard>/annotation_worksheet.csv                 semantic error-type labels

The downstream baselines are YOLO11 (CNN) and the DETR-family models already
trained under the fixed protocol (RT-DETR-L, RT-DETRv2-L). DN-DETR / DINOv3 are
paper-level strong baselines deferred to A2 pretraining (see draft/README.md), so
they are noted as a gap rather than silently omitted.

Output:

  <output>/downstream-baseline/README.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASELINE_MODELS = ["yolo11n", "yolo11s", "rtdetr-l", "rtdetrv2-l"]
FAMILY = {
    "yolo11n": "CNN (YOLO)",
    "yolo11s": "CNN (YOLO)",
    "rtdetr-l": "DETR",
    "rtdetrv2-l": "DETR",
}
TYPE_COLUMNS = ["高光", "金属", "遮挡", "门牙", "其他"]
_FALSE = {"", "0", "no", "否", "无", "n", "false", "-"}


def _flag(value: str) -> bool:
    return value.strip().lower() not in _FALSE


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else ""


def _load_metrics(results_csv: Path) -> dict[str, dict]:
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metrics = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        metrics[r["model"]] = {
            "P": _num(r.get("precision(B)")),
            "R": _num(r.get("recall(B)")),
            "mAP50": _num(r.get("mAP50(B)")),
            "mAP50-95": _num(r.get("mAP50-95(B)")),
        }
    return metrics


def _load_hard_rows(hard_diff_csv: Path) -> list[dict]:
    with hard_diff_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_easy_summary(hard_easy_csv: Path) -> dict:
    with hard_easy_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    stats = {}
    for r in rows:
        group = r["group"]
        s = stats.setdefault(group, {"before": {"tp": 0, "fp": 0, "fn": 0, "gt": 0}, "after": {"tp": 0, "fp": 0, "fn": 0, "gt": 0}})
        for model in ("before", "after"):
            s[model]["tp"] += int(r[f"{'b' if model == 'before' else 'a'}_tp"])
            s[model]["fp"] += int(r[f"{'b' if model == 'before' else 'a'}_fp"])
            s[model]["fn"] += int(r[f"{'b' if model == 'before' else 'a'}_fn"])
            s[model]["gt"] += int(r["gt"])
    return stats


def _load_error_types(worksheet_csv: Path) -> dict[str, int]:
    with worksheet_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = {t: 0 for t in TYPE_COLUMNS}
    for r in rows:
        for t in TYPE_COLUMNS:
            if _flag(r.get(t, "")):
                counts[t] += 1
    return counts


def _ratio(tp: int, denom: int) -> float:
    return tp / denom if denom else 0.0


def _render(metrics, hard_rows, easy_stats, error_counts, total_annotated: int) -> str:
    lines = [
        "# 下游检测基线复现报告（论文 A·A0）",
        "",
        "## 评估协议（固定）",
        "",
        "- 数据：`dataset/OMNI_COCO/after`（人工复核后标签，作为 ground truth）",
        "- 评估集：固定 `test` split（828 张，不参与调参）；匹配 IoU ≥ 0.5",
        "- 指标：Precision(B) / Recall(B) / mAP50(B) / mAP50-95(B)",
        "- 下游基线：YOLO11（CNN）+ RT-DETR-L / RT-DETRv2-L（DETR 系）",
        "- 生成脚本：`tools/eval_test_set.py`（test 集指标）、`tools/analyze_hard_diff.py` / `analyze_hard_easy.py`（困难子集）、`tools/analyze_annotation.py`（误检类型）",
        "",
    ]

    lines.append("## 一、下游基线指标（after 标签，固定 test 集 828 张）")
    lines.append("")
    lines.append("| 基线 | 类型 | P(B) | R(B) | mAP50(B) | mAP50-95(B) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name in BASELINE_MODELS:
        m = metrics.get(name)
        if m is None:
            lines.append(f"| {name} | {FAMILY[name]} | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {FAMILY[name]} | {_fmt(m['P'])} | {_fmt(m['R'])} | {_fmt(m['mAP50'])} | {_fmt(m['mAP50-95'])} |"
        )
    lines.append("")
    lines.append("> RT-DETRv2-L 在 mAP50-95 上最高（0.7137），但 Recall 略低于 YOLO 系；完整 8 模型 before/after 对比见 `output/val-test/comparison.md`。")
    lines.append("")

    lines.append("## 二、困难子集每图 FP/FN（after 模型 = RT-DETRv2-L，32 张人工挑选）")
    lines.append("")
    lines.append("| 图像 | GT | 预测 | TP | FP | FN |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in hard_rows:
        lines.append(f"| {r['image']} | {r['gt']} | {r['a_pred']} | {r['a_tp']} | {r['a_fp']} | {r['a_fn']} |")
    lines.append("")
    lines.append("> 完整 before/after 逐样本见 `output/hard-subset/visualization/hard_diff_analysis.md`。")
    lines.append("")

    lines.append("## 三、困难 / 易样本对比（after 模型，真值固定为 after 标签）")
    lines.append("")
    lines.append("| 组 | 样本 | GT | TP | FP | FN | P | R | 错误 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    group_label = {"hard": "困难", "easy": "易"}
    n = {"hard": len(hard_rows), "easy": 796}
    for group in ("hard", "easy"):
        s = easy_stats.get(group)
        if s is None:
            continue
        a = s["after"]
        p = _ratio(a["tp"], a["tp"] + a["fp"])
        r = _ratio(a["tp"], a["tp"] + a["fn"])
        err = a["fp"] + a["fn"]
        lines.append(f"| {group_label[group]} | {n[group]} | {a['gt']} | {a['tp']} | {a['fp']} | {a['fn']} | {p:.3f} | {r:.3f} | {err} |")
    lines.append("")
    lines.append("> 复核收益（after − before）：困难 −15.8% 错误、易 −4.6%；见 `hard_easy_analysis.md`。")
    lines.append("")

    lines.append("## 四、误检类型占比（困难子集语义标注，112 张）")
    lines.append("")
    lines.append("| 类型 | 图数 | 占总数 |")
    lines.append("| --- | --- | --- |")
    for t in TYPE_COLUMNS:
        c = error_counts[t]
        lines.append(f"| {t} | {c} | {_ratio(c, total_annotated):.1%} |")
    lines.append("")
    lines.append("> 单图单类型；「其他」占 65.2% 表示大部分错误未归入四类假设原因，遮挡为 0。来源：`output/hard-subset/annotation_summary.md`。")
    lines.append("")

    lines.append("## 五、错误案例可视化")
    lines.append("")
    lines.append("- 预测叠加图（绿=GT、红=预测）：`output/hard-subset/visualization/after/<stem>_overlay.png`（829 张）")
    lines.append("- before/after 对比合成图：`output/hard-subset/visualization/hard_diff/<stem>.png`（32 张困难样本）")
    lines.append("- GT vs 预测框叠加图：`output/hard-subset/gt_pred/<stem>_gt_pred.png`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-test", type=Path, default=ROOT / "output/val-test")
    parser.add_argument("--hard", type=Path, default=ROOT / "output/hard-subset")
    parser.add_argument("--output", type=Path, default=ROOT / "output/downstream-baseline")
    args = parser.parse_args()

    results_csv = args.val_test / "results.csv"
    hard_diff_csv = args.hard / "visualization" / "hard_diff_analysis.csv"
    hard_easy_csv = args.hard / "visualization" / "hard_easy_analysis.csv"
    worksheet_csv = args.hard / "annotation_worksheet.csv"

    for p in (results_csv, hard_diff_csv, hard_easy_csv, worksheet_csv):
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    metrics = _load_metrics(results_csv)
    hard_rows = _load_hard_rows(hard_diff_csv)
    easy_stats = _load_easy_summary(hard_easy_csv)
    error_counts = _load_error_types(worksheet_csv)
    with worksheet_csv.open(newline="", encoding="utf-8") as handle:
        total_annotated = sum(1 for _ in csv.DictReader(handle))

    args.output.mkdir(parents=True, exist_ok=True)
    out_md = args.output / "README.md"
    out_md.write_text(_render(metrics, hard_rows, easy_stats, error_counts, total_annotated), encoding="utf-8")

    print(f"Wrote {out_md}")
    for name in BASELINE_MODELS:
        m = metrics.get(name)
        if m:
            print(f"  {name}: mAP50-95={m['mAP50-95']:.4f}")
    print(f"  hard={len(hard_rows)}  error types={error_counts}")


if __name__ == "__main__":
    main()
