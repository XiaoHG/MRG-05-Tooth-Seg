"""Build an Excel annotation worksheet with embedded overlay images.

CSV cannot embed raster images, so this turns ``annotation_worksheet.csv`` into
``annotation_worksheet.xlsx`` where each row carries a thumbnail of the model's
``<stem>_overlay.png`` next to the editable type columns (高光/金属/遮挡/门牙/其他)
and a free-text 备注 column.

Input (produced by ``tools/build_annotation_worksheet.py`` + a predict-dir run):

  <input>/annotation_worksheet.csv   one row per candidate image
  <input>/visualization/<stem>/<stem>_overlay.png   per-image prediction overlay

Output:

  <output>/annotation_worksheet.xlsx
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]

# Column layout: keep the CSV order, inserting an embedded-image column after 图像.
HEADERS = [
    "序号", "图像", "可视化", "overlay", "子集", "GT", "预测", "TP", "FP", "FN",
    "错误", "亮度", "高光占比", "高光", "金属", "遮挡", "门牙", "其他", "备注",
]
IMAGE_COL = 3  # "可视化" (1-indexed)

INT_COLS = {"序号", "GT", "预测", "TP", "FP", "FN", "错误"}
FLOAT_COLS = {"亮度", "高光占比"}
EDITABLE_COLS = {"高光", "金属", "遮挡", "门牙", "其他", "备注"}

# Column widths (Excel character units); the image column is sized to the
# 512px-wide overlay thumbnails (~73 units).
COLUMN_WIDTHS = {
    "序号": 6, "图像": 18, "可视化": 78, "overlay": 52, "子集": 10,
    "GT": 6, "预测": 6, "TP": 6, "FP": 6, "FN": 6, "错误": 6,
    "亮度": 9, "高光占比": 11, "高光": 6, "金属": 6, "遮挡": 6, "门牙": 6, "其他": 6, "备注": 30,
}
ROW_HEIGHT = 265  # enough for a 341px-tall overlay (341px * 0.75 = 256pt)


def _typed(key: str, value: str):
    if key in INT_COLS:
        return int(value)
    if key in FLOAT_COLS:
        return float(value)
    return value


def _overlay_path(vis_dir: Path, stem: str) -> Path:
    return vis_dir / stem / f"{stem}_overlay.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/hard-subset")
    parser.add_argument("--output", type=Path, default=ROOT / "output/hard-subset")
    args = parser.parse_args()

    csv_path = args.input / "annotation_worksheet.csv"
    vis_dir = args.input / "visualization"
    out_path = args.output / "annotation_worksheet.xlsx"

    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    wb = Workbook()
    ws = wb.active
    ws.title = "标注工作表"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="4472C4")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(horizontal="left", vertical="center", wrap_text=True)

    # Header row.
    for col_idx, name in enumerate(HEADERS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        ws.column_dimensions[get_column_letter(col_idx)].width = COLUMN_WIDTHS[name]
    ws.row_dimensions[1].height = 22

    missing: list[str] = []
    for index, r in enumerate(rows, start=1):
        row_idx = index + 1
        stem = r["图像"]
        for col_idx, name in enumerate(HEADERS, start=1):
            if name == "可视化":
                continue  # filled by the embedded image below
            value = _typed(name, r[name]) if name in r else ""
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.alignment = center if name not in {"备注", "overlay"} else left

        overlay = _overlay_path(vis_dir, stem)
        if not overlay.is_file():
            missing.append(stem)
            continue
        img = XLImage(str(overlay))
        ws.add_image(img, f"C{row_idx}")
        ws.row_dimensions[row_idx].height = ROW_HEIGHT

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(HEADERS))}{len(rows) + 1}"

    wb.save(out_path)

    print(f"Wrote {out_path} ({len(rows)} rows)")
    if missing:
        print(f"WARNING: {len(missing)} overlays missing: {missing[:20]}")


if __name__ == "__main__":
    main()
