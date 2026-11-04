"""Sync human annotations from the Excel worksheet back into the CSV.

The reviewer edits ``annotation_worksheet.xlsx`` (which embeds overlay images),
not the CSV. This script copies the filled semantic columns
(高光 / 金属 / 遮挡 / 门牙 / 其他 / 备注) from the XLSX back into
``annotation_worksheet.csv`` so the CSV stays the canonical machine-readable
record — and so re-running ``build_annotation_xlsx.py`` does not lose the
annotations.

Input:

  <dir>/annotation_worksheet.xlsx   human-filled sheet
  <dir>/annotation_worksheet.csv    fill-in sheet (semantic cols may be blank)

Output:

  <dir>/annotation_worksheet.csv    semantic cols updated in place (a .csv.bak is kept)
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC = ["高光", "金属", "遮挡", "门牙", "其他", "备注"]


def _read_xlsx(path: Path) -> dict[str, dict[str, str]]:
    wb = load_workbook(path, data_only=True)
    ws = wb.active
    headers = [c.value for c in ws[1]]
    col = {name: i for i, name in enumerate(headers) if name}
    img_col = col["图像"]
    sem_cols = {name: col[name] for name in SEMANTIC}

    annot: dict[str, dict[str, str]] = {}
    for r in range(2, ws.max_row + 1):
        stem = ws.cell(row=r, column=img_col + 1).value
        if stem is None:
            continue
        annot[str(stem)] = {
            name: _cell(ws, r, idx) for name, idx in sem_cols.items()
        }
    return annot


def _cell(ws, row: int, col_idx: int) -> str:
    value = ws.cell(row=row, column=col_idx + 1).value
    return "" if value is None else str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=ROOT / "output/hard-subset")
    args = parser.parse_args()

    xlsx_path = args.dir / "annotation_worksheet.xlsx"
    csv_path = args.dir / "annotation_worksheet.csv"
    if not xlsx_path.exists():
        raise SystemExit(f"{xlsx_path} not found")
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found")

    annot = _read_xlsx(xlsx_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0
    missing: list[str] = []
    for row in rows:
        stem = row.get("图像", "")
        if stem in annot:
            for name in SEMANTIC:
                row[name] = annot[stem][name]
            updated += 1
        else:
            missing.append(stem)

    backup = csv_path.with_name(csv_path.name + ".bak")
    shutil.copy2(csv_path, backup)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Synced {updated} rows from {xlsx_path.name} -> {csv_path.name}")
    print(f"Backup: {backup.name}")
    if missing:
        print(f"WARNING: {len(missing)} CSV rows not in xlsx: {missing[:20]}")


if __name__ == "__main__":
    main()
