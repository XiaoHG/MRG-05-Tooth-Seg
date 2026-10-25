# -*- coding: utf-8 -*-
"""Convert optimized chapter .md files into .docx matching the GPT-5.6-Terra-medium format.

Reference format:
  - Page: US Letter, margins top/bottom 1", left/right 1.25"
  - Font: SimSun (宋体) throughout
  - Title (#): 16pt bold, centered
  - Section heading (##) / subsection (###): 12pt bold
  - Body: 12pt, first-line indent 0.5", 1.5 line spacing
  - Keywords paragraph: no indent
  - Tables: Table Grid style, 10pt cells
  - Equations (lines containing a lambda): centered, no indent
"""
import os
import re

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

FONT = "宋体"
BODY_SIZE = 12
TITLE_SIZE = 16
TABLE_SIZE = 10

BASE = os.path.dirname(os.path.abspath(__file__))
MD_DIR = os.path.abspath(os.path.join(BASE, ".."))


def _set_font(run, size=BODY_SIZE, bold=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), FONT)
    rfonts.set(qn("w:hAnsi"), FONT)
    rfonts.set(qn("w:eastAsia"), FONT)
    rfonts.set(qn("w:cs"), FONT)


def _setup_page(doc):
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(1)
    sec.bottom_margin = Inches(1)
    sec.left_margin = Inches(1.25)
    sec.right_margin = Inches(1.25)
    sec.header_distance = Inches(0.5)
    sec.footer_distance = Inches(0.5)


def _add_title(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    _set_font(run, size=TITLE_SIZE, bold=True)


def _add_heading(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.5
    run = p.add_run(text)
    _set_font(run, size=BODY_SIZE, bold=True)


def _add_runs_with_bold(p, text, size=BODY_SIZE):
    """Parse **bold** inline markers and add runs."""
    parts = re.split(r"\*\*(.+?)\*\*", text)
    for i, part in enumerate(parts):
        if part == "":
            continue
        bold = (i % 2 == 1)
        run = p.add_run(part)
        _set_font(run, size=size, bold=bold)


def _add_body(doc, text, indent=True, align="justify"):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if align == "justify" else WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    if indent:
        pf.first_line_indent = Inches(0.5)
    _add_runs_with_bold(p, text)
    return p


def _add_equation(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(4)
    pf.space_after = Pt(4)
    run = p.add_run(text)
    _set_font(run, size=BODY_SIZE)


def _add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.line_spacing = 1.2
    run = p.add_run(text)
    _set_font(run, size=10.5, bold=True)


def _add_image(doc, rel_path):
    full = os.path.join(BASE, rel_path)
    if os.path.exists(full):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(full, width=Inches(5.5))
    else:
        p = doc.add_paragraph()
        run = p.add_run("[缺失图片: %s]" % rel_path)
        _set_font(run, size=BODY_SIZE)


def _set_cell_font(cell, text, bold=False):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    _set_font(run, size=TABLE_SIZE, bold=bold)


def _add_table(doc, header, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        _set_cell_font(table.rows[0].cells[j], h, bold=True)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            _set_cell_font(table.rows[i + 1].cells[j], val, bold=False)
    # trailing spacing after table
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(6)


def _parse_table(lines):
    """Parse a markdown pipe table. Returns (header, rows) or None."""
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    # skip separator line
    body_lines = [l for l in lines[1:] if not re.match(r"^\s*\|?[\s:|-]+\|?\s*$", l)]
    rows = []
    for l in body_lines:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        rows.append(cells)
    return header, rows


def convert(md_path, out_path):
    with open(md_path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    doc = Document()
    # default style font
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(BODY_SIZE)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    _setup_page(doc)

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped == "":
            i += 1
            continue

        # title
        if stripped.startswith("# "):
            _add_title(doc, stripped[2:].strip())
            i += 1
            continue

        # headings
        if stripped.startswith("### "):
            _add_heading(doc, stripped[4:].strip())
            i += 1
            continue
        if stripped.startswith("## "):
            _add_heading(doc, stripped[3:].strip())
            i += 1
            continue

        # image
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if m:
            caption, rel = m.group(1), m.group(2)
            _add_image(doc, rel)
            _add_caption(doc, caption)
            i += 1
            continue

        # table (current line starts with | and next line is a separator)
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1].strip()):
            block = [stripped]
            i += 1
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            header, rows = _parse_table(block)
            _add_table(doc, header, rows)
            continue

        # equation (line contains a lambda)
        if "λ" in stripped or stripped.startswith("L_"):
            _add_equation(doc, stripped)
            i += 1
            continue

        # keywords paragraph (no indent)
        plain = stripped.replace("**", "")
        if plain.startswith("关键词") or plain.startswith("Keywords"):
            _add_body(doc, plain, indent=False)
            i += 1
            continue

        # normal body paragraph
        _add_body(doc, stripped)
        i += 1

    doc.save(out_path)
    return out_path


CHAPTERS = [
    "chapter_omni_coco-abstract",
    "chapter_omni_coco-introduction",
    "chapter_omni_coco-related_work",
    "chapter_omni_coco-method",
    "chapter_omni_coco-experiment",
    "chapter_omni_coco-conclusion",
    "chapter_omni_coco-manual_review",
    "chapter_omni_coco-references",
]


def main():
    for name in CHAPTERS:
        md_path = os.path.join(MD_DIR, name + ".md")
        out_path = os.path.join(BASE, name + ".docx")
        if not os.path.exists(md_path):
            print("SKIP (no md):", md_path)
            continue
        convert(md_path, out_path)
        print("OK:", os.path.basename(out_path))


if __name__ == "__main__":
    main()
