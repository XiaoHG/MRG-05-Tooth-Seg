"""Build a human annotation worksheet from the hard-subset screen output.

Input (produced by ``tools/hard_subset_screen.py``):

  <input>/screen.csv     one row per image with per-image tp/fp/fn + features
  <input>/errors.json    per-image FP/FN box details

The worksheet splits the candidates into three *mutually exclusive* groups so a
reviewer can batch the work by error type:

  * fp-subset (误检):  fp >= 3 and fn < 3  -> top N by fp desc
  * fn-subset (漏检):  fn >= 3 and fp < 3  -> top N by fn desc
  * mixed (整图失败):  fp >= 3 and fn >= 3 -> all, by (fp+fn) desc

Semantic labels (glare / metal / occlusion / incisor / other) are left blank for
the reviewer to fill in. This script only produces the *list*; it does not decide
the semantic cause of any error.

Output:

  <output>/annotation_worksheet.csv   the fill-in sheet (one row per image)
  <output>/annotation_worksheet.md    a grouped, human-readable copy + instructions
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUBSET_MIXED = "整图失败"
SUBSET_FP = "误检子集"
SUBSET_FN = "漏检子集"

TYPE_COLUMNS = ["高光", "金属", "遮挡", "门牙", "其他"]


def _int(row: dict, key: str) -> int:
    return int(row[key])


def _float(row: dict, key: str) -> float:
    return float(row[key])


def _build_groups(rows: list[dict], top_n: int) -> dict[str, list[dict]]:
    mixed = [r for r in rows if _int(r, "fp") >= 3 and _int(r, "fn") >= 3]
    fp_only = [r for r in rows if _int(r, "fp") >= 3 and _int(r, "fn") < 3]
    fn_only = [r for r in rows if _int(r, "fn") >= 3 and _int(r, "fp") < 3]

    mixed.sort(key=lambda r: (-(_int(r, "fp") + _int(r, "fn")), -_int(r, "fp")))
    fp_only.sort(key=lambda r: (-_int(r, "fp"), -_int(r, "fn")))
    fn_only.sort(key=lambda r: (-_int(r, "fn"), -_int(r, "fp")))

    return {
        SUBSET_MIXED: mixed,
        SUBSET_FP: fp_only[:top_n],
        SUBSET_FN: fn_only[:top_n],
    }


def _csv_row(r: dict, subset: str, index: int) -> dict:
    return {
        "序号": index,
        "图像": r["image"],
        "子集": subset,
        "GT": r["gt_count"],
        "预测": r["pred_count"],
        "TP": r["tp"],
        "FP": r["fp"],
        "FN": r["fn"],
        "错误": r["error"],
        "亮度": r["brightness"],
        "高光占比": r["highlight_ratio"],
        "高光": "",
        "金属": "",
        "遮挡": "",
        "门牙": "",
        "其他": "",
        "备注": "",
    }


def _csv_fieldnames() -> list[str]:
    return [
        "序号", "图像", "子集", "GT", "预测", "TP", "FP", "FN", "错误",
        "亮度", "高光占比", *TYPE_COLUMNS, "备注",
    ]


def _markdown(groups: dict[str, list[dict]]) -> str:
    total = sum(len(v) for v in groups.values())
    lines = [
        "# 困难子集标注工作表",
        "",
        "本表由 `tools/build_annotation_worksheet.py` 从初筛结果自动生成，",
        "把候选图分成三组（互斥），便于按错误类型分批人工复核。",
        "",
        "## 使用说明",
        "",
        "1. 打开图像（`dataset/OMNI_COCO/after/images/test/<图像>.jpg` 等），对照 `errors.json` 中的 FP/FN 框明细逐张查看。",
        "2. 在 `高光 / 金属 / 遮挡 / 门牙 / 其他` 对应列填 `1`（一张图可同时命中多个类型）。",
        "3. `备注` 可补充具体位置（如“上颌左中切牙误检”“下颌第二磨牙漏检”）。",
        "4. 完成后把填好的 `annotation_worksheet.csv` 存档，作为 v13 困难子集的人工确认结果。",
        "",
        f"- 候选总数：**{total}**（误检 {len(groups[SUBSET_FP])} + 漏检 {len(groups[SUBSET_FN])} + 整图失败 {len(groups[SUBSET_MIXED])}）",
        "",
        "> 类型含义：高光=牙面反光/过曝；金属=金属修复体/正畸托槽等；遮挡=邻牙/唇颊/器械遮挡；门牙=切牙区漏检；其他=以上之外的困难原因。",
        "",
    ]
    for subset in (SUBSET_MIXED, SUBSET_FP, SUBSET_FN):
        rows = groups[subset]
        lines.append(f"## {subset}（{len(rows)} 张）")
        lines.append("")
        lines.append(
            "| 序号 | 图像 | GT | 预测 | TP | FP | FN | 错误 | 亮度 | 高光占比 | 高光 | 金属 | 遮挡 | 门牙 | 其他 | 备注 |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for index, r in enumerate(rows, start=1):
            lines.append(
                f"| {index} | {r['image']} | {r['gt_count']} | {r['pred_count']} | {r['tp']} "
                f"| {r['fp']} | {r['fn']} | {r['error']} | {r['brightness']} | {r['highlight_ratio']} "
                f"|  |  |  |  |  |  |"
            )
        lines.append("")
    lines.append("FP/FN 框明细见 `errors.json`；初筛全量见 `screen.csv`。")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/hard-subset")
    parser.add_argument("--output", type=Path, default=ROOT / "output/hard-subset")
    parser.add_argument("--top", type=int, default=50, help="Max images per fp/fn subset.")
    args = parser.parse_args()

    screen_csv = args.input / "screen.csv"
    if not screen_csv.exists():
        raise SystemExit(f"screen.csv not found: {screen_csv}")

    with screen_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    groups = _build_groups(rows, args.top)

    args.output.mkdir(parents=True, exist_ok=True)

    out_csv = args.output / "annotation_worksheet.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames())
        writer.writeheader()
        index = 0
        for subset in (SUBSET_MIXED, SUBSET_FP, SUBSET_FN):
            for r in groups[subset]:
                index += 1
                writer.writerow(_csv_row(r, subset, index))

    out_md = args.output / "annotation_worksheet.md"
    out_md.write_text(_markdown(groups), encoding="utf-8")

    total = sum(len(v) for v in groups.values())
    print(
        f"Wrote {out_csv} and {out_md}\n"
        f"整图失败={len(groups[SUBSET_MIXED])} "
        f"误检={len(groups[SUBSET_FP])} "
        f"漏检={len(groups[SUBSET_FN])} "
        f"共 {total} 张"
    )


if __name__ == "__main__":
    main()
