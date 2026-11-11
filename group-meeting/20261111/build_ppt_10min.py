#!/usr/bin/env python3
"""Build the 10-minute 2026-11-11 group-meeting deck for OralDINO.

Run with the venv that has python-pptx installed:

    /tmp/pptx-env/bin/python group-meeting/20261111/build_ppt.py

Output: OralDINO_group_meeting_20261111_10min.pptx in the same directory.
"""

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
OUT = HERE / "OralDINO_group_meeting_20261111_10min.pptx"

# ---------------------------------------------------------------- palette
NAVY = RGBColor(0x0B, 0x25, 0x45)
TEAL = RGBColor(0x1B, 0x99, 0x8B)
ORANGE = RGBColor(0xF4, 0xA2, 0x61)
INK = RGBColor(0x1F, 0x29, 0x37)
GRAY = RGBColor(0x6B, 0x72, 0x80)
LIGHT = RGBColor(0xF4, 0xF7, 0xFA)
PALE = RGBColor(0xE8, 0xF2, 0xF0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x2A, 0x9D, 0x8F)
AMBER = RGBColor(0xB7, 0x79, 0x1A)
BORDER = RGBColor(0xE2, 0xE8, 0xF0)
FONT = "Arial"
EA_FONT = "PingFang SC"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


# ---------------------------------------------------------------- helpers
def style_run(run, size=None, bold=None, color=None, font=FONT):
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    run.font.name = font
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", EA_FONT)


def tx(slide, x, y, w, h, text, size=16, bold=False, color=INK,
       align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.1):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = anchor
    lines = str(text).split("\n")
    for j, line in enumerate(lines):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = line
        style_run(r, size, bold, color)
    return box


def bullets(slide, x, y, w, h, items, size=15, color=INK, spacing=7,
            bullet_color=TEAL, line_spacing=1.12):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(spacing)
        p.line_spacing = line_spacing
        r0 = p.add_run()
        r0.text = "•  "
        style_run(r0, size, True, bullet_color)
        if "：" in item:
            head, tail = item.split("：", 1)
            r1 = p.add_run()
            r1.text = head + "："
            style_run(r1, size, True, color)
            r2 = p.add_run()
            r2.text = tail
            style_run(r2, size, False, color)
        else:
            r1 = p.add_run()
            r1.text = item
            style_run(r1, size, False, color)
    return box


def card(slide, x, y, w, h, title, body, accent=TEAL, fill=LIGHT,
         title_size=17, body_size=12.5):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = BORDER
    shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    try:
        shape.adjustments[0] = 0.06
    except Exception:
        pass
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.14), Inches(0.07), Inches(h - 0.28)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.fill.background()
    tx(slide, x + 0.22, y + 0.12, w - 0.32, 0.34, title,
       size=title_size, bold=True, color=NAVY)
    tx(slide, x + 0.22, y + 0.52, w - 0.36, h - 0.6, body,
       size=body_size, color=INK, line_spacing=1.14)
    return shape


def table(slide, x, y, w, h, data, col_widths=None, font_size=12,
          header_fill=NAVY, header_color=WHITE, band=True,
          highlight_rows=None, aligns=None):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = shape.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Inches(cw)
    for r in range(rows):
        for c in range(cols):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            if aligns and c < len(aligns):
                p.alignment = aligns[c]
            elif c == 0:
                p.alignment = PP_ALIGN.LEFT
            else:
                p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(data[r][c])
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
                style_run(run, font_size, True, header_color)
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = (
                    WHITE if (not band or r % 2 == 1) else RGBColor(0xF3, 0xF6, 0xFA)
                )
                style_run(run, font_size, False, INK)
    if highlight_rows:
        for r, color in highlight_rows.items():
            for c in range(cols):
                cell = tbl.cell(r, c)
                cell.fill.solid()
                cell.fill.fore_color.rgb = color
    return tbl


def status_color(tbl, col, mapping):
    for r, color in mapping.items():
        p = tbl.cell(r, col).text_frame.paragraphs[0]
        for run in p.runs:
            run.font.color.rgb = color
            run.font.bold = True


def footer(slide, idx):
    tx(slide, 0.6, 7.08, 8.0, 0.24, "OralDINO · 组会报告",
       size=9, color=GRAY)
    tx(slide, 11.6, 7.08, 1.1, 0.24, str(idx), size=9, color=GRAY,
       align=PP_ALIGN.RIGHT)


def section(slide, text):
    tx(slide, 0.6, 0.26, 9.0, 0.26, text, size=11, bold=True, color=TEAL)


def title(slide, text, sub=None):
    tx(slide, 0.6, 0.54, 12.1, 0.62, text, size=26, bold=True, color=NAVY)
    if sub:
        tx(slide, 0.6, 1.13, 12.1, 0.34, text if False else sub, size=13, color=GRAY)


def blank():
    return prs.slides.add_slide(prs.slide_layouts[6])


def arrow(slide, x, y, w=0.45, h=0.30, color=TEAL):
    shp = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def box(slide, x, y, w, h, text, fill=PALE, line=TEAL, color=NAVY,
        size=13, bold=True):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = line
    shp.line.width = Pt(1.0)
    shp.shadow.inherit = False
    try:
        shp.adjustments[0] = 0.08
    except Exception:
        pass
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for j, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        style_run(r, size, bold, color)
    return shp


# ================================================================ Slide 1
s = blank()
s.background.fill.solid()
s.background.fill.fore_color.rgb = NAVY
shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(1.75),
                         Inches(0.12), Inches(2.55))
shp.fill.solid()
shp.fill.fore_color.rgb = TEAL
shp.line.fill.background()
tx(s, 1.25, 1.65, 11.2, 0.4, "组会进度汇报", size=13, color=TEAL, bold=True)
tx(s, 1.25, 2.05, 11.2, 0.95, "OralDINO", size=54, bold=True, color=WHITE)
tx(s, 1.25, 3.00, 11.2, 0.55,
   "多源口腔影像的 DINOv3 域内预训练与下游验证", size=23,
   color=RGBColor(0xD6, 0xE4, 0xF0))
tx(s, 1.25, 3.72, 11.2, 0.4,
   "牙齿检测 · 氟斑牙分级 · 口腔疾病筛查", size=15,
   color=RGBColor(0xA8, 0xBE, 0xD4))
tx(s, 1.25, 5.55, 11.2, 0.35,
   "数据审计  →  域内预训练  →  多任务下游评估", size=12,
   color=RGBColor(0x8F, 0xA6, 0xBF))

# ================================================================ Slide 2
s = blank()
section(s, "01 / 背景与问题")
title(s, "口腔筛查的可及性、域鸿沟与标注瓶颈")
bullets(s, 0.7, 1.5, 7.0, 3.7, [
    "临床需求：口腔疾病筛查依赖专业医生，基层与偏远地区可及性不足。",
    "低成本入口：智能手机口内照片，但存在高光、金属、遮挡与视角变化。",
    "域鸿沟：通用 DINOv3 在自然图像预训练，口内影像存在明显域偏移。",
    "数据瓶颈：多源牙科数据分散、标准不一；氟斑牙 4 级仅 141 张。",
], size=14)
rqs = [
    ("RQ1  域内预训练", "多源口内图能否得到\n可迁移的 OralDINO？"),
    ("RQ2  下游有效性", "相对 base DINOv3，氟斑牙 /\n口腔疾病 / 检测是否提升？"),
    ("RQ3  适配策略", "冻结 / LoRA / 全量，\n哪种在小数据上最优？"),
]
for i, (t, b) in enumerate(rqs):
    card(s, 8.0, 1.5 + i * 1.35, 4.55, 1.15, t, b,
         accent=TEAL, title_size=13, body_size=11)
route = ["多源数据", "审计去重", "OralDINO", "多任务评估", "论文"]
for i, label in enumerate(route):
    x = 0.75 + i * 2.42
    box(s, x, 5.85, 1.95, 0.75, label, fill=PALE if i < 4 else NAVY,
        line=TEAL if i < 4 else NAVY, color=NAVY if i < 4 else WHITE, size=12)
    if i < len(route) - 1:
        arrow(s, x + 2.0, 6.07, 0.35, 0.28)
footer(s, 2)

# ================================================================ Slide 3
s = blank()
section(s, "01 / 创新点")
title(s, "三个创新点与总体路线")
cards = [
    ("01  数据层", "多源口内数据整合、审计与去重\n55,559 文件 → 42,618 唯一图\n坏图清理 + 评测泄漏检查"),
    ("02  预训练层", "小 batch 下的 DINOv3 域内预训练\nsoftmax + center 替换 Sinkhorn\n官方 recipe + 224→512 渐进"),
    ("03  评估层", "氟斑牙 4 级 + 口腔疾病 6 类\n牙齿检测 A3\n冻结 / LoRA / 全量"),
]
for i, (t, b) in enumerate(cards):
    card(s, 0.7 + i * 4.02, 1.5, 3.78, 2.2, t, b,
         accent=[TEAL, ORANGE, NAVY][i], title_size=16, body_size=12)
stages = [
    ("A0", "数据/基线"), ("A1", "数据整合"), ("A2", "域内预训练"),
    ("A3", "下游验证"), ("A4", "撰写"),
]
for i, (a, label) in enumerate(stages):
    x = 0.7 + i * 2.43
    fill = RGBColor(0xFF, 0xF1, 0xE2) if a == "A3" else PALE
    line = ORANGE if a == "A3" else TEAL
    box(s, x, 4.0, 2.0, 0.7, f"{a}  {label}", fill=fill, line=line,
        color=NAVY, size=11.5)
    if i < len(stages) - 1:
        arrow(s, x + 2.03, 4.21, 0.34, 0.28)
card(s, 0.7, 4.95, 11.75, 1.85, "讨论点",
     "数据质量与多样性：大规模图像训练需要控制质量与多样性，这两者是生成优质特征的要素。\n"
     "DINOv3 相关研究表明：来自小数据集的非常高质量数据组成的同质批次是有益的。\n"
     "对 DINOv3 进行预训练（或微调）的调优。",
     accent=ORANGE, title_size=14, body_size=10.5)
footer(s, 3)

# ================================================================ Slide 4 (DINOv3)
s = blank()
section(s, "01 / 方法背景")
title(s, "对 DINOv3 的理解：自监督 + 师生 EMA + 多裁剪")
box(s, 0.6, 1.3, 1.6, 0.6, "输入图像", size=10.5)
arrow(s, 2.25, 1.46, 0.3, 0.26)
box(s, 2.65, 1.3, 3.2, 0.6, "多裁剪\n2 global 224 + 6 local 96", size=9.5)
arrow(s, 5.9, 1.46, 0.3, 0.26)
box(s, 6.3, 1.3, 1.3, 0.6, "ViT-B/16", fill=NAVY, line=NAVY, color=WHITE, size=10.5)
box(s, 0.6, 2.2, 3.3, 0.85, "Student ViT-B/16\n12 blocks · RoPE · 4 storage tokens",
    size=9, fill=PALE, line=TEAL)
box(s, 4.2, 2.2, 3.3, 0.85, "Teacher ViT-B/16（EMA）\n不接收梯度",
    size=9, fill=RGBColor(0xE8, 0xEE, 0xF7), line=NAVY)
arrow(s, 3.95, 2.48, 0.25, 0.24)
tx(s, 3.9, 2.02, 0.4, 0.2, "EMA", size=7.5, color=GRAY)
for x in (1.4, 3.5):
    shp = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(x), Inches(3.02),
                             Inches(0.22), Inches(0.28))
    shp.fill.solid()
    shp.fill.fore_color.rgb = TEAL
    shp.line.fill.background()
    shp.shadow.inherit = False
box(s, 0.6, 3.35, 1.9, 0.7, "DINO head（CLS）\n32768 prototypes", size=9)
box(s, 2.7, 3.35, 1.9, 0.7, "iBOT head（patch）\nmasked patch 预测", size=9)
box(s, 4.8, 3.35, 2.7, 0.7, "Teacher targets\nsoftmax + center",
    size=9, fill=RGBColor(0xE8, 0xEE, 0xF7), line=NAVY)
box(s, 0.6, 4.35, 6.9, 0.85,
    "Loss = DINO CLS + iBOT（masked patch） + KoLeo\n学生预测教师分布；teacher 由 EMA 提供稳定目标",
    size=9, fill=RGBColor(0xFF, 0xF1, 0xE2), line=ORANGE)
tx(s, 0.6, 5.35, 6.9, 0.9,
   "我们的改动：softmax+center 替换 Sinkhorn；32768 prototypes；Gram 关闭；224/96 → 512/128。",
   size=9.5, color=GRAY, line_spacing=1.1)
bullets(s, 8.0, 1.3, 4.6, 5.7, [
    "自监督：不需要标签；多裁剪提供尺度/视角。",
    "师生 EMA：teacher 提供稳定目标，避免表示坍缩。",
    "iBOT：掩码 patch 预测，学习局部细节。",
    "RoPE + 4 storage tokens：支持可变分辨率。",
    "DINO head：32768 原型，形成语义聚类。",
    "我们：softmax+center、K=32768、Gram off、224→512。",
], size=10.5)
footer(s, 4)

# ================================================================ Slide 4
s = blank()
section(s, "01 / 方法")
title(s, "多源数据与预训练设计")
data = [
    ["来源", "文件", "唯一"],
    ["AlphaDent", "1,455", "1,441"],
    ["COde", "49,938", "37,020"],
    ["OMNI", "4,166", "4,157"],
    ["合计", "55,559", "42,618"],
]
table(s, 0.7, 1.65, 4.5, 2.6, data,
      col_widths=[1.7, 1.35, 1.35], font_size=12,
      highlight_rows={4: RGBColor(0xE8, 0xF2, 0xF0)})
tx(s, 0.7, 4.4, 4.6, 1.0,
   "SHA256 去重；删除 2 张截断图；\n97.3% 的图短边 < 384。",
   size=10.5, color=GRAY)
bullets(s, 5.6, 1.55, 6.8, 3.3, [
    "DINOv3 ViT-B/16；2 global 224 + 6 local 96；student/teacher EMA。",
    "损失：DINO CLS + iBOT patch + KoLeo；Gram 关闭。",
    "Sinkhorn 小 batch 退化 → softmax + center。",
    "32768 prototypes；batch 320；224/96 → 512/128。",
], size=13.5)
data2 = [
    ["阶段", "分辨率", "batch", "步数", "峰值 LR", "显存"],
    ["Stage 0", "224/96", "96", "2,220", "1.22e-4", "7.9 GB"],
    ["Stage 1", "224/96", "320", "16,080", "1.21e-4", "21.6 GB"],
    ["Stage 2", "512/128", "24", "3,216", "~1.3e-5 起", "未实测"],
]
table(s, 0.7, 5.15, 11.75, 1.55, data2,
      col_widths=[1.3, 1.3, 1.0, 1.2, 1.5, 1.3], font_size=11)
footer(s, 5)

# ================================================================ Slide 5
s = blank()
section(s, "01 / 架构与下游")
title(s, "模型架构与下游评估")
box(s, 0.7, 2.15, 2.35, 0.9, "多源口内图像\n42,616 张", size=11.5)
arrow(s, 3.15, 2.47, 0.38, 0.28)
box(s, 3.65, 2.15, 2.45, 0.9, "OralDINO\nViT-B/16", fill=NAVY, line=NAVY,
    color=WHITE, size=12.5)
for y in (1.94, 3.29, 4.64):
    arrow(s, 6.2, y, 0.9, 0.28)
branches = [
    ("氟斑牙 4 级", "141 张；QWK / F1 / Acc"),
    ("口腔疾病 6 类", "720 张；Acc / F1"),
    ("牙齿检测 A3", "828 固定 test；mAP50-95"),
]
for i, (t, b) in enumerate(branches):
    card(s, 7.3, 1.5 + i * 1.35, 5.1, 1.15, t, b,
         accent=[TEAL, ORANGE, NAVY][i], title_size=13, body_size=11)
shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(5.75),
                         Inches(11.75), Inches(0.8))
shp.fill.solid()
shp.fill.fore_color.rgb = RGBColor(0xFF, 0xF6, 0xEC)
shp.line.color.rgb = ORANGE
shp.shadow.inherit = False
tx(s, 0.9, 5.84, 11.35, 0.62,
   "术语：A3 = 论文的下游有效性评估阶段（A0 数据/基线 → A1 数据 → A2 预训练 → "
   "A3 下游验证 → A4 撰写）；「检测 A3」= 用牙齿检测任务验证 OralDINO。",
   size=10.5, color=NAVY, line_spacing=1.08)
footer(s, 6)

# ================================================================ Slide 6
s = blank()
section(s, "01 / 实验设计")
title(s, "数据、评估协议与实验矩阵")
data = [
    ["用途", "数据", "规模"],
    ["预训练", "AlphaDent + COde + OMNI", "42,618 唯一图"],
    ["氟斑牙", "4 级分级", "141 张"],
    ["口腔疾病", "公开 6 类", "720 张"],
    ["牙齿检测", "OMNI_COCO 固定 test", "828 张"],
]
table(s, 0.7, 1.65, 5.0, 2.6, data,
      col_widths=[1.2, 2.4, 1.4], font_size=11.5)
bullets(s, 0.7, 4.45, 5.0, 1.3, [
    "分类：冻结 CLS + Logistic Regression，5 折 × 5 次。",
    "适配：冻结 / LoRA / 全量；检测：固定 test。",
], size=11)
data2 = [
    ["编号", "实验", "状态"],
    ["E1", "检测基线 8 模型", "完成"],
    ["E2", "P0 42k 预训练", "失败"],
    ["E3", "Stage 0 诊断", "完成"],
    ["E4", "Stage 1 主训练", "44%"],
    ["E5", "Stage 2 高分辨率", "待做"],
    ["E6", "冻结 probe", "待做"],
    ["E7", "LoRA / 全量", "待做"],
    ["E8", "检测 A3", "待做"],
    ["E9", "消融", "待做"],
]
tbl = table(s, 5.95, 1.65, 6.5, 3.55, data2,
            col_widths=[0.7, 4.4, 1.3], font_size=11,
            aligns=[PP_ALIGN.CENTER, PP_ALIGN.LEFT, PP_ALIGN.CENTER])
status_color(tbl, 2, {1: GREEN, 2: RED, 3: GREEN, 4: AMBER, 5: GRAY,
                      6: GRAY, 7: GRAY, 8: GRAY, 9: GRAY})
shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(5.65),
                         Inches(11.75), Inches(0.72))
shp.fill.solid()
shp.fill.fore_color.rgb = RGBColor(0xFF, 0xF1, 0xE2)
shp.line.color.rgb = ORANGE
shp.shadow.inherit = False
tx(s, 0.95, 5.78, 11.3, 0.5,
   "关键 gate：先做 E6 冻结 probe；如果 7k/10k/16k 都不优于 base，暂停 LoRA / 检测 A3 的大投入。",
   size=11.5, bold=True, color=NAVY)
footer(s, 7)

# ================================================================ Slide 9
s = blank()
section(s, "02 / 实验进度")
title(s, "Stage 1 完成：5k→16k 全部低于 base")
card(s, 0.7, 1.5, 4.2, 2.05, "Stage 1 完成（16,080 步）",
     "16,080 / 16,080；batch 320\n"
     "DINO 0.345 / 0.263；iBOT 2.352\n"
     "KoLeo -0.252；21.6 GB\n"
     "最终 ckpt 15,999",
     accent=GREEN, title_size=14, body_size=11.5)
card(s, 0.7, 3.75, 4.2, 1.55, "检查点趋势（QWK）",
     "5k 0.847 → 10k 0.872\n"
     "15k 0.886 → 16k 0.884\n"
     "训练越久越好；早停不能救",
     accent=RED, title_size=14, body_size=11)
tx(s, 0.7, 5.55, 4.3, 0.9,
   "Stage 0：2,220 步，DINO 10.40→9.68，证明 softmax 生效。",
   size=10.5, color=GRAY)
data = [
    ["模型", "氟斑牙 QWK（200）", "口腔疾病 QWK（720）", "口腔疾病 Acc"],
    ["base", "0.916", "0.848", "0.828"],
    ["5k teacher", "0.847", "0.808", "0.795"],
    ["10k teacher", "0.872", "0.819", "0.799"],
    ["15k teacher", "0.886", "0.820", "0.799"],
    ["16k teacher", "0.884", "0.820", "0.799"],
]
tbl = table(s, 5.25, 1.55, 7.3, 2.55, data,
            col_widths=[2.0, 1.8, 1.8, 1.7], font_size=11,
            highlight_rows={1: RGBColor(0xE8, 0xF2, 0xF0)})
bullets(s, 5.25, 4.35, 7.3, 1.9, [
    "student 与 teacher 趋势一致（差 ≤0.005）；全部低于 base，p<0.001。",
    "5k 已过度分散：pair cosine 0.16；15k/16k 仍 0.14。",
    "下一步：base 蒸馏/冻结 Gram、降 K/KoLeo；不再等早停。",
], size=11)
footer(s, 8)

# ================================================================ Slide 10
s = blank()
section(s, "03 / 下一步")
title(s, "问题、下一步与论文安排")
card(s, 0.7, 1.5, 3.75, 4.7, "当前问题",
     "早停假设否定：5k 最差，15k 最好\n\n"
     "过度分散在 5k 已发生（0.16）\n\n"
     "全参数 recipe 不能超过 base\n\n"
     "K/N=0.77；数据源偏置",
     accent=RED, title_size=15, body_size=11.5)
card(s, 4.75, 1.5, 3.75, 4.7, "下一步",
     "base 蒸馏 / 冻结 Gram anchor\n\n"
     "降 K/KoLeo/LR，短 schedule\n\n"
     "数据质量/来源平衡\n\n"
     "LoRA/全量验证；检测 A3 后置",
     accent=TEAL, title_size=15, body_size=11.5)
card(s, 8.8, 1.5, 3.65, 4.7, "论文安排",
     "11 月：配方迭代 + LoRA\n\n"
     "12 月：下游适配 + 检测 A3\n\n"
     "1 月：初稿\n\n"
     "2 月：投稿\n\n"
     "目标会议 / 期刊：待定",
     accent=NAVY, title_size=15, body_size=11.5)
tx(s, 0.7, 6.4, 11.75, 0.4,
   "原则：先解决特征漂移，再投入下游大实验。",
   size=11.5, bold=True, color=GRAY)
footer(s, 9)

# ================================================================ Slide 11
s = blank()
section(s, "03 / 论文")
title(s, "论文摘要（草稿）")
tx(s, 0.7, 1.18, 11.75, 0.32,
   "OralDINO: Domain-Adaptive DINOv3 Pretraining for Intraoral Images",
   size=12, bold=True, color=GRAY)
card(s, 0.7, 1.62, 11.75, 4.85, "摘要",
     "口腔疾病智能筛查高度依赖精细化口内影像特征建模，然而口腔影像存在标注资源稀缺、"
     "采集环境异质性强等固有问题，通用视觉预训练模型与口腔专属影像域之间存在显著域偏移，"
     "导致通用模型在口腔临床任务中泛化能力受限。此外，现有公开口腔影像数据集存在数据源分散、"
     "标注标准不统一、高质量预训练样本匮乏等缺陷，缺乏适配医学自监督预训练的标准化数据资源与"
     "系统性评估体系。\n"
     "针对上述问题，本文构建了一套面向口腔影像的域自适应自监督预训练框架。首先，本文整合 "
     "AlphaDent、COde、OMNI 等多源公开口内影像数据集，通过 SHA256 哈希去重与多层级样本质量"
     "审计机制，筛选得到 42,616 张高质量唯一口内影像，构建了标准化口腔无标注预训练数据集。\n"
     "基于 DINOv3 ViT-B/16 原生师生自监督学习架构，开展口腔影像域持续预训练，有效缩小通用"
     "视觉模型与口腔影像域的域鸿沟。针对原生 Sinkhorn-Knopp 算法在小批次训练场景下易出现"
     "教师伪标签均匀退化、训练监督信号失效的关键问题，本文引入 Softmax+Center 训练稳定策略，"
     "有效规避伪标签坍缩问题，保障自监督训练过程稳定收敛。同时，本文创新性设计了低分辨率"
     "初始化、高分辨率渐进适配的训练范式，通过 224/96 基础分辨率完成全局与局部特征粗粒度学习，"
     "再切换至 512/128 高分辨率实现口腔细微病灶特征精细化建模，适配口腔疾病细粒度识别需求。\n"
     "为系统性验证所提预训练框架的有效性，本文构建包含氟斑牙四级有序分级、口腔疾病六分类、"
     "牙齿目标检测在内的多维度下游评测任务体系，采用固定数据集划分、多随机种子实验机制，"
     "以 QWK、Macro-F1、mAP50-95 作为统一量化指标，全面对比冻结线性探测、LoRA 微调、全量"
     "微调三种迁移学习范式，并以原生 DINOv3 模型作为基线对照。\n"
     "目前 Stage 1 已完成；5k/10k/15k/16k 的冻结线性 probe 在氟斑牙 4 级与口腔疾病 6 类上"
     "均未超过 base DINOv3，且 5k 起已出现特征过度分散。下一步将评估 base 蒸馏/正则化配方"
     "与 LoRA/全量微调，最终结论待补充。",
     accent=TEAL, title_size=15, body_size=13)
tx(s, 0.7, 6.62, 11.75, 0.3,
   "关键词：口腔影像 · DINOv3 · 自监督预训练 · 域自适应 · 氟斑牙分级 · 牙齿检测",
   size=10.5, color=GRAY)
footer(s, 10)

# ================================================================ Slide 12
s = blank()
section(s, "总结")
title(s, "总结与讨论")
tx(s, 0.7, 1.5, 5.9, 0.35, "三点结论", size=15, bold=True, color=TEAL)
bullets(s, 0.7, 1.95, 5.9, 3.1, [
    "待解决：pair cosine 0.82→0.14 / 0.63→0.07，特征过度分散，语义聚类被破坏。",
    "P0 失败已归因：loss 下降不等于下游提升；官方 recipe + softmax 是当前路线。",
    "5k/10k/15k/16k 全部低于 base，早停假设否定；下一步改配方（base 蒸馏/降 K）与 LoRA/全量。",
], size=13)
tx(s, 6.95, 1.5, 5.5, 0.35, "三个请教问题", size=15, bold=True, color=ORANGE)
bullets(s, 6.95, 1.95, 5.5, 3.1, [
    "论文的目标会议 / 期刊与时间线是否合适？",
    "下游优先级：氟斑牙分级，还是其他口腔疾病检测？",
    "氟斑牙数据是否继续扩量 / 补标注，还是先用 200 张做方法验证？",
], size=13, bullet_color=ORANGE)
shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(5.55),
                         Inches(11.75), Inches(0.85))
shp.fill.solid()
shp.fill.fore_color.rgb = NAVY
shp.line.fill.background()
shp.shadow.inherit = False
tx(s, 0.95, 5.78, 11.3, 0.45, "谢谢，欢迎讨论。",
   size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
footer(s, 11)

prs.save(OUT)
prs.save(HERE / "覃建海-组会20261111.pptx")
print(f"saved: {OUT}")
print(f"saved: {HERE / '覃建海-组会20261111.pptx'}")
print(f"slides: {len(prs.slides)}")
