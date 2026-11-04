"""Summarise the human-filled semantic annotation worksheet.

Input (produced by ``tools/build_annotation_worksheet.py``, then filled in by a
reviewer):

  <input>/annotation_worksheet.csv   one row per candidate image; the semantic
                                     columns 高光/金属/遮挡/门牙/其他 are filled
                                     with ``1`` where the error type applies.

This script *reads back* the filled sheet and reports the error-type proportions
(误检类型占比) needed by v13 task 3. It does not decide any semantic label; it only
aggregates what the reviewer has marked.

Output:

  <output>/annotation_summary.md    per-type counts / proportions + per-subset matrix

The script degrades gracefully when the sheet is still blank: it reports how many
rows have been annotated and lists the unannotated ones, so progress is verifiable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TYPE_COLUMNS = ["高光", "金属", "遮挡", "门牙", "其他"]
SUBSET_ORDER = ["整图失败", "误检子集", "漏检子集"]

# Values that mean "not flagged" (blank cells are also treated as empty).
_FALSE_VALUES = {"", "0", "no", "否", "无", "n", "false", "-"}


def _flagged(value: str) -> bool:
    return value.strip().lower() not in _FALSE_VALUES


def _pct(n: int, denom: int) -> str:
    return f"{n / denom:.1%}" if denom else "—"


def _subset_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {subset: {t: 0 for t in TYPE_COLUMNS} for subset in SUBSET_ORDER}
    for r in rows:
        subset = r.get("子集", "")
        if subset not in matrix:
            continue
        for t in TYPE_COLUMNS:
            if _flagged(r.get(t, "")):
                matrix[subset][t] += 1
    return matrix


def _int(row: dict, key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, ValueError, TypeError):
        return 0


def _summarise(rows: list[dict]) -> dict:
    total = len(rows)
    annotated = [r for r in rows if any(_flagged(r.get(t, "")) for t in TYPE_COLUMNS)]
    unannotated = [r for r in rows if not any(_flagged(r.get(t, "")) for t in TYPE_COLUMNS)]

    per_type: dict[str, dict] = {}
    for t in TYPE_COLUMNS:
        flagged_rows = [r for r in rows if _flagged(r.get(t, ""))]
        per_type[t] = {
            "n": len(flagged_rows),
            "sum_fp": sum(_int(r, "FP") for r in flagged_rows),
            "sum_fn": sum(_int(r, "FN") for r in flagged_rows),
            "sum_error": sum(_int(r, "错误") for r in flagged_rows),
        }

    matrix = _subset_counts(rows)
    return {
        "total": total,
        "annotated": annotated,
        "unannotated": unannotated,
        "per_type": per_type,
        "matrix": matrix,
    }


def _markdown(s: dict, total_error: int) -> str:
    lines = [
        "# 困难子集语义标注汇总",
        "",
        f"- 候选图像总数：**{s['total']}**（`annotation_worksheet.csv`）",
        f"- 已标注：**{len(s['annotated'])}** / {s['total']}；未标注：**{len(s['unannotated'])}**",
        f"- 语义类型：高光 / 金属 / 遮挡 / 门牙 / 其他（一张图可同时命中多个类型，故占比之和可超过 100%）",
        "",
    ]

    if not s["annotated"]:
        lines.append("> ⚠ 尚未填写任何语义标签，请先人工对照 overlay 图像完成标注。")
        lines.append("")
        return "\n".join(lines)

    lines.append("## 误检类型占比（按图，多标签）")
    lines.append("")
    lines.append("| 类型 | 图数 | 占总数 | 涉及 FP | 涉及 FN | 涉及错误(FP+FN) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for t in TYPE_COLUMNS:
        d = s["per_type"][t]
        lines.append(
            f"| {t} | {d['n']} | {_pct(d['n'], s['total'])} | {d['sum_fp']} | {d['sum_fn']} | {d['sum_error']} |"
        )
    lines.append("")
    lines.append(f"> “涉及”= 被标记为该类型的图上的 FP/FN 求和（多标签，跨类型会重复计入）。全表 FP+FN 合计参考 `hard_easy_analysis.md`。")
    lines.append("")

    lines.append("## 按子集 × 类型")
    lines.append("")
    header = "| 类型 | " + " | ".join(SUBSET_ORDER) + " | 合计 |"
    lines.append(header)
    lines.append("| --- | " + " | ".join(["---"] * (len(SUBSET_ORDER) + 1)) + " |")
    for t in TYPE_COLUMNS:
        counts = [s["matrix"][sub][t] for sub in SUBSET_ORDER]
        lines.append(f"| {t} | " + " | ".join(str(c) for c in counts) + f" | {sum(counts)} |")
    lines.append("")

    if s["unannotated"]:
        preview = "、".join(r.get("图像", "?") for r in s["unannotated"][:20])
        more = "" if len(s["unannotated"]) <= 20 else f" 等共 {len(s['unannotated'])} 张"
        lines.append(f"## 未标注图像（{len(s['unannotated'])} 张）")
        lines.append("")
        lines.append(f"{preview}{more}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/hard-subset")
    parser.add_argument("--output", type=Path, default=ROOT / "output/hard-subset")
    args = parser.parse_args()

    csv_path = args.input / "annotation_worksheet.csv"
    if not csv_path.exists():
        raise SystemExit(f"annotation_worksheet.csv not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    s = _summarise(rows)
    total_error = sum(_int(r, "错误") for r in rows)

    out_md = args.output / "annotation_summary.md"
    out_md.write_text(_markdown(s, total_error), encoding="utf-8")

    print(
        f"Wrote {out_md}\n"
        f"total={s['total']} annotated={len(s['annotated'])} unannotated={len(s['unannotated'])} "
        f"total_error={total_error}"
    )
    for t in TYPE_COLUMNS:
        d = s["per_type"][t]
        print(f"  {t}: n={d['n']} fp={d['sum_fp']} fn={d['sum_fn']} err={d['sum_error']}")


if __name__ == "__main__":
    main()
