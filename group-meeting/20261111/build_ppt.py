#!/usr/bin/env python3
"""Build the 2026-11-11 group-meeting deck for OralDINO.

Run with the venv that has python-pptx installed:

    /tmp/pptx-env/bin/python group-meeting/20261111/build_ppt.py

Output: OralDINO_group_meeting_20261111.pptx in the same directory.
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
OUT = HERE / "OralDINO_group_meeting_20261111.pptx"

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
section(s, "01 / 汇报提纲")
title(s, "三部分：创新点与设计、实验进度、下一步计划")
cards = [
    ("01  创新点与实验设计",
     "研究问题与总体路线\n三个创新点\n预训练与下游架构\n实验矩阵与评估协议"),
    ("02  实验进度",
     "检测基线：8 模型 × 828 test\nP0 失败复盘与归因\n官方 DINOv3 Stage 0 / Stage 1\n当前问题与风险"),
    ("03  下一步计划",
     "续训与 checkpoint 导出\n冻结 probe / LoRA / 检测 A3\n消融实验\n论文撰写安排"),
]
for i, (t, b) in enumerate(cards):
    card(s, 0.7 + i * 4.12, 1.8, 3.82, 4.55, t, b,
         accent=[TEAL, ORANGE, NAVY][i], title_size=18, body_size=14)
footer(s, 2)

# ================================================================ Slide 3
s = blank()
section(s, "01 / 研究背景")
title(s, "口腔筛查的可及性与标注瓶颈")
bullets(s, 0.7, 1.55, 7.55, 4.9, [
    "临床需求：口腔疾病筛查依赖专业医生，基层与偏远地区可及性不足。",
    "低成本入口：智能手机口内照片，但存在高光、金属、遮挡与视角变化。",
    "任务链：牙齿检测 → 实例分割 / FDI 牙位 → 龋病 / 氟斑牙分级。",
    "标注瓶颈：氟斑牙 4 级仅 141 张；检测框标注成本高、跨库标准不一致。",
    "技术机会：自监督预训练可利用无标注口内图像，但通用 DINOv3 存在域鸿沟。",
], size=15)
stats = [("42,616", "唯一预训练图"), ("141", "氟斑牙评估图"),
         ("828", "固定检测 test"), ("8", "检测基线模型")]
for i, (num, label) in enumerate(stats):
    x = 8.65 + (i % 2) * 2.05
    y = 1.65 + (i // 2) * 2.35
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                             Inches(1.9), Inches(1.95))
    shp.fill.solid()
    shp.fill.fore_color.rgb = LIGHT
    shp.line.color.rgb = BORDER
    shp.shadow.inherit = False
    tx(s, x, y + 0.35, 1.9, 0.55, num, size=28, bold=True, color=TEAL,
       align=PP_ALIGN.CENTER)
    tx(s, x, y + 1.05, 1.9, 0.45, label, size=12, color=GRAY,
       align=PP_ALIGN.CENTER)
footer(s, 3)

# ================================================================ Slide 4
s = blank()
section(s, "01 / 相关工作")
title(s, "相关工作与空白")
items = [
    ("DINOv3", "自监督 ViT；dense token 表征；LVD-1689M 自然图像预训练。"),
    ("DinoDental", "评估自然图像 DINOv3 的牙科迁移；不做牙科域内预训练。"),
    ("多源牙科数据", "AlphaDent / COde / OMNI；口内照片为主，采集与标注标准不一。"),
    ("牙科检测方法", "TLNM / Mask R-CNN；Caries-DETR；以逐实例预测为主。"),
]
for i, (t, b) in enumerate(items):
    x = 0.7 + (i % 2) * 6.0
    y = 1.55 + (i // 2) * 2.05
    card(s, x, y, 5.75, 1.8, t, b, accent=[TEAL, ORANGE, NAVY, GRAY][i],
         title_size=16, body_size=12.5)
shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(5.85),
                         Inches(11.75), Inches(0.85))
shp.fill.solid()
shp.fill.fore_color.rgb = NAVY
shp.line.fill.background()
shp.shadow.inherit = False
tx(s, 0.95, 6.02, 11.3, 0.5,
   "研究空白：缺少“多源口内数据 + DINOv3 域内预训练 + 统一下游评估”的完整闭环。",
   size=14, bold=True, color=WHITE)
footer(s, 4)

# ================================================================ Slide 5
s = blank()
section(s, "01 / 研究问题")
title(s, "研究问题与总体技术路线")
rqs = [
    ("RQ1  域内预训练", "多源口内图能否通过 DINOv3 continue-pretraining\n得到可迁移的 OralDINO backbone？"),
    ("RQ2  下游有效性", "相对 base DINOv3，氟斑牙分级、口腔疾病分类\n与牙齿检测是否取得提升？"),
    ("RQ3  适配策略", "冻结线性 probe、LoRA、全量微调，\n哪种策略在口腔小数据上最优？"),
]
for i, (t, b) in enumerate(rqs):
    card(s, 0.7 + i * 4.0, 1.55, 3.75, 2.0, t, b, accent=TEAL,
         title_size=15, body_size=11.5)
route = ["多源口内数据", "审计 / 去重", "OralDINO", "多任务评估", "论文"]
for i, label in enumerate(route):
    x = 0.75 + i * 2.42
    box(s, x, 4.55, 1.95, 0.85, label, fill=PALE if i < 4 else NAVY,
        line=TEAL if i < 4 else NAVY, color=NAVY if i < 4 else WHITE, size=13)
    if i < len(route) - 1:
        arrow(s, x + 2.0, 4.82, 0.35, 0.30)
tx(s, 0.75, 5.75, 11.6, 0.65,
   "主线下游：氟斑牙 4 级分级 + 其他口腔疾病分类 + 牙齿检测 A3；所有实验共用固定 split 与统一指标。",
   size=12.5, color=GRAY)
footer(s, 5)

# ================================================================ Slide 6
s = blank()
section(s, "01 / 创新点")
title(s, "三个创新点：数据、预训练、评估")
cards = [
    ("01  数据层", "多源口内数据整合、审计与去重\n55,559 文件 → 42,618 唯一图\n坏图清理 + 评测泄漏检查"),
    ("02  预训练层", "小 batch 下的 DINOv3 域内预训练\nsoftmax + center 替换 Sinkhorn\n官方 recipe + 224→512 渐进"),
    ("03  评估层", "面向多任务的统一下游协议\n氟斑牙 + 口腔疾病 + 检测\n冻结 / LoRA / 全量三种适配"),
]
for i, (t, b) in enumerate(cards):
    card(s, 0.7 + i * 4.02, 1.65, 3.78, 3.55, t, b,
         accent=[TEAL, ORANGE, NAVY][i], title_size=17, body_size=13)
tx(s, 0.7, 5.55, 11.75, 0.7,
   "论文主线：把“数据 → 预训练 → 下游”闭环做扎实；不在一篇论文里叠加过多模块。",
   size=13, bold=True, color=NAVY)
footer(s, 6)

# ================================================================ Slide 7
s = blank()
section(s, "01 / 创新点一")
title(s, "多源数据整合、审计与去重")
bullets(s, 0.7, 1.6, 6.85, 4.7, [
    "整合 AlphaDent、COde、OMNI 三个公开口内数据集。",
    "SHA256 去重：55,559 个文件 → 42,618 张唯一图。",
    "删除 2 张截断 JPEG；与评测集哈希重叠为 0。",
    "分辨率审计：中位 448×298；97.3% 的图短边 < 384。",
    "先审计再训练：避免重复、坏图与评测泄漏。",
], size=15)
data = [
    ["来源", "文件", "唯一"],
    ["AlphaDent", "1,455", "1,441"],
    ["COde-Dataset", "49,938", "37,020"],
    ["omni_coco", "4,166", "4,157"],
    ["合计", "55,559", "42,618"],
]
tbl = table(s, 7.85, 1.7, 4.55, 3.4, data,
            col_widths=[1.95, 1.3, 1.3], font_size=13,
            highlight_rows={4: RGBColor(0xE8, 0xF2, 0xF0)})
tx(s, 7.85, 5.35, 4.55, 0.9,
   "去重视图用硬链接落盘，训练时直接指向 dataset/pretrain_dedup。",
   size=11.5, color=GRAY)
footer(s, 7)

# ================================================================ Slide 8
s = blank()
section(s, "01 / 创新点二")
title(s, "小 batch 下的 DINOv3 域内预训练")
card(s, 0.7, 1.55, 5.75, 4.35, "问题：Sinkhorn 在小 batch 下退化",
     "batch 8 时 teacher 目标只有 16 行 × 65536 原型，\n"
     "行列归一化把 logits 抹平 → 目标接近均匀。\n\n"
     "DINO loss ≈ ln(K)，55k 步不下降；\n"
     "梯度方向是“最大化 student 输出熵”，\n"
     "反而在把 CLS 头往均匀分布推。",
     accent=RED, title_size=16, body_size=12.5)
card(s, 6.75, 1.55, 5.7, 4.35, "方案：softmax + center + 官方 recipe",
     "用 DINOv2 风格 softmax + center 替换 Sinkhorn；\n"
     "官方 DINOv3：ViT-B/16 + iBOT + KoLeo；\n"
     "32768 prototypes；batch 320；\n"
     "224/96 主训练 → 512/128 低 LR 适应。\n\n"
     "Stage 0 验证：DINO loss 10.40 → 9.68。",
     accent=TEAL, title_size=16, body_size=12.5)
tx(s, 0.7, 6.15, 11.75, 0.55,
   "这是从 P0 失败中得到的关键方法改动：先把目标函数修对，再谈训练量。",
   size=13, bold=True, color=NAVY)
footer(s, 8)

# ================================================================ Slide 9
s = blank()
section(s, "01 / 创新点三")
title(s, "多任务下游评估协议")
cards = [
    ("氟斑牙 4 级", "normal / mild / moderate / severe\n141 张评估图（P0 口径）\nQWK / Macro-F1 / Accuracy\n轻度与中度边界是难点"),
    ("口腔疾病 6 类", "720 张公开数据\nAccuracy / Macro-F1\n检验通用口腔表征"),
    ("牙齿检测 A3", "828 张固定 test\nmAP50 / mAP50-95\nRT-DETRv2-L / YOLO26m"),
]
for i, (t, b) in enumerate(cards):
    card(s, 0.7 + i * 4.02, 1.55, 3.78, 2.65, t, b,
         accent=[TEAL, ORANGE, NAVY][i], title_size=16, body_size=12.5)
bullets(s, 0.7, 4.55, 11.75, 1.2, [
    "适配策略：冻结线性 probe / LoRA / 全量微调，三种一起做可比。",
    "统一协议：同一数据划分、同一指标、多种子重复。",
    "checkpoint 选择：以下游 probe 为准，不用 training loss 选模型。",
], size=14)
shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(5.95),
                         Inches(11.75), Inches(0.78))
shp.fill.solid()
shp.fill.fore_color.rgb = RGBColor(0xFF, 0xF6, 0xEC)
shp.line.color.rgb = ORANGE
shp.shadow.inherit = False
tx(s, 0.9, 6.02, 11.35, 0.64,
   "术语：A3 = 论文的下游有效性评估阶段（A0 数据/基线 → A1 数据整合 → "
   "A2 域内预训练 → A3 下游验证 → A4 撰写）；「检测 A3」= 用牙齿检测任务验证 OralDINO。",
   size=10.5, color=NAVY, line_spacing=1.08)
footer(s, 9)

# ================================================================ Slide 10
s = blank()
section(s, "01 / 模型架构")
title(s, "模型架构总图")
box(s, 0.65, 2.35, 2.55, 1.05, "多源口内图像\n42,616 张", size=12)
arrow(s, 3.30, 2.72, 0.40, 0.30)
box(s, 3.80, 2.35, 2.35, 1.05, "数据审计\nSHA256 去重", size=12)
arrow(s, 6.25, 2.72, 0.40, 0.30)
box(s, 6.75, 2.15, 2.55, 1.45, "OralDINO\nViT-B/16\nDINOv3", fill=NAVY,
    line=NAVY, color=WHITE, size=13)
arrow(s, 9.40, 2.72, 0.40, 0.30)
# downstream branches
branches = [
    ("冻结 / LoRA / 全量", "氟斑牙 4 级\n口腔疾病 6 类"),
    ("检测 A3（下游验证）", "RT-DETRv2-L / YOLO26m\nmAP50 / mAP50-95"),
    ("统一评估", "QWK / F1 / Acc / mAP\n固定 split + 多种子"),
]
for i, (t, b) in enumerate(branches):
    y = 1.25 + i * 1.75
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.95), Inches(y),
                             Inches(2.75), Inches(1.35))
    shp.fill.solid()
    shp.fill.fore_color.rgb = LIGHT
    shp.line.color.rgb = BORDER
    shp.shadow.inherit = False
    tx(s, 10.10, y + 0.10, 2.45, 0.30, t, size=12.5, bold=True, color=NAVY)
    tx(s, 10.10, y + 0.45, 2.45, 0.75, b, size=10.5, color=INK, line_spacing=1.12)
tx(s, 0.7, 5.75, 11.75, 0.75,
   "预训练只做一次，下游共享同一个 backbone；每条下游都保留 base DINOv3 作为对照。",
   size=13, color=GRAY)
footer(s, 10)

# ================================================================ Slide 11
s = blank()
section(s, "01 / 预训练设计")
title(s, "预训练关键设计")
bullets(s, 0.7, 1.55, 6.55, 4.6, [
    "2 个 global crop（224）+ 6 个 local crop（96）；student / teacher EMA。",
    "损失：DINO CLS + iBOT patch + KoLeo；Gram 暂时关闭。",
    "softmax + center：解决小 batch 下 teacher 目标退化。",
    "32768 prototypes；K/N ≈ 0.77，存在头部记忆风险，需 probe 验证。",
    "先 224 主训练，再 512 短适应：数据中位分辨率仅 448×298。",
], size=14.5)
data = [
    ["阶段", "分辨率", "batch", "步数", "峰值 LR", "显存"],
    ["Stage 0", "224/96", "96", "2,220", "1.22e-4", "7.9 GB"],
    ["Stage 1", "224/96", "320", "16,080", "1.21e-4", "21.6 GB"],
    ["Stage 2", "512/128", "24", "3,216", "~1.3e-5 起", "未实测"],
]
table(s, 7.55, 1.7, 5.0, 2.55, data,
      col_widths=[0.95, 0.95, 0.65, 0.8, 1.0, 0.85], font_size=11.5)
tx(s, 7.55, 4.5, 5.0, 1.5,
   "Stage 1 的 16,080 步 = 120 epochs × 134 steps；\n"
   "Stage 2 续训 3,216 步，保持同一 schedule 单位。",
   size=11.5, color=GRAY)
tx(s, 0.7, 6.15, 11.75, 0.55,
   "关键顺序：先修目标函数（softmax），再做显存标定（batch 320），最后才拉长训练。",
   size=13, bold=True, color=NAVY)
footer(s, 11)

# ================================================================ Slide 12
s = blank()
section(s, "01 / 数据与评估")
title(s, "数据与评估协议")
data = [
    ["用途", "数据", "规模"],
    ["预训练", "AlphaDent + COde + OMNI", "42,618 唯一图"],
    ["氟斑牙", "4 级分级", "141 张（P0 口径）"],
    ["口腔疾病", "公开 6 类", "720 张"],
    ["牙齿检测", "OMNI_COCO 固定 test", "828 张"],
]
table(s, 0.7, 1.65, 6.1, 2.55, data,
      col_widths=[1.25, 2.85, 2.0], font_size=12.5)
tx(s, 0.7, 4.45, 6.1, 1.4,
   "氟斑牙 4 级：normal / mild / moderate / severe；\n"
   "P0 评估集 141 张（历史 5 类含 doubtful，4 级协议待重跑）。",
   size=11.5, color=GRAY)
bullets(s, 7.05, 1.65, 5.55, 4.4, [
    "分类：冻结 CLS + 标准化 + Logistic Regression，5 折 × 5 次。",
    "适配：LoRA / 全量微调，与冻结 probe 对比。",
    "检测：mAP50 / mAP50-95，固定 test，不参与调参。",
    "指标：QWK / Macro-F1 / Acc / mAP。",
    "报告：均值 ± 标准差，多种子重复。",
], size=13.5)
footer(s, 12)

# ================================================================ Slide 13
s = blank()
section(s, "01 / 实验矩阵")
title(s, "实验矩阵与当前状态")
data = [
    ["编号", "实验", "状态"],
    ["E1", "检测基线 8 模型（828 固定 test）", "完成"],
    ["E2", "P0 42k 预训练（batch 8 / Sinkhorn）", "失败，已归因"],
    ["E3", "Stage 0 诊断（224/96，2,220 步）", "完成"],
    ["E4", "Stage 1 主训练（batch 320，16,080 步）", "完成"],
    ["E5", "Stage 2 高分辨率适应（512/128）", "待做"],
    ["E6", "冻结线性 probe（5k/10k/15k/16k）", "完成"],
    ["E7", "LoRA / 全量微调（vs base）", "待做"],
    ["E8", "检测 A3（RT-DETRv2-L / YOLO26m）", "待做"],
    ["E9", "消融：K / batch / KoLeo / 分辨率 / Gram", "待做"],
]
tbl = table(s, 0.7, 1.55, 8.3, 5.0, data,
            col_widths=[0.8, 5.8, 1.7], font_size=12,
            aligns=[PP_ALIGN.CENTER, PP_ALIGN.LEFT, PP_ALIGN.CENTER])
status_color(tbl, 2, {
    1: GREEN, 2: RED, 3: GREEN, 4: GREEN, 5: GRAY,
    6: GREEN, 7: GRAY, 8: GRAY, 9: GRAY,
})
card(s, 9.35, 1.75, 3.15, 3.9, "关键 gate",
     "probe 已完成：15k/16k 均低于 base。\n\n"
     "下一步先评 5k/10k；\n如果仍不优于 base，\n转 LoRA/全量或调配方。",
     accent=ORANGE, title_size=15, body_size=12)
footer(s, 13)

# ================================================================ Slide 14
s = blank()
section(s, "02 / 实验进度")
title(s, "检测基线：8 模型 × 828 固定 test")
data = [
    ["模型", "before mAP50-95", "after mAP50-95", "Δ", "after mAP50"],
    ["RT-DETRv2-L", "0.6697", "0.7137", "+0.0440", "0.8749"],
    ["Faster R-CNN", "0.6326", "0.6583", "+0.0256", "0.8250"],
    ["RT-DETR-L", "0.6036", "0.6372", "+0.0337", "0.9845"],
    ["YOLO26m", "0.6037", "0.6333", "+0.0296", "0.9867"],
    ["YOLO11n", "0.6003", "0.6324", "+0.0321", "0.9855"],
    ["YOLOv8m", "0.5986", "0.6324", "+0.0338", "0.9857"],
    ["YOLO11s", "0.6008", "0.6320", "+0.0312", "0.9852"],
    ["RetinaNet", "0.5491", "0.5715", "+0.0224", "0.7191"],
]
tbl = table(s, 0.7, 1.55, 8.4, 4.65, data,
            col_widths=[1.8, 1.7, 1.7, 1.2, 1.6], font_size=12,
            highlight_rows={1: RGBColor(0xFF, 0xF1, 0xE2)})
bullets(s, 9.35, 1.65, 3.35, 4.6, [
    "8 个模型的 mAP50-95 全部提升，Δ +0.022 ~ +0.044。",
    "收益集中在严格定位质量，mAP50 已经饱和。",
    "RT-DETRv2-L 对标注质量最敏感：+0.044 / Recall +0.062。",
], size=12.5)
tx(s, 0.7, 6.45, 11.75, 0.4,
   "结论：检测基线可复现、排名稳定；A3 用 RT-DETRv2-L 与 YOLO26m 作为主基线。",
   size=12.5, bold=True, color=NAVY)
footer(s, 14)

# ================================================================ Slide 15
s = blank()
section(s, "02 / 实验进度")
title(s, "P0 失败复盘：loss 下降 ≠ 下游提升")
data = [
    ["指标", "base", "P0 10k", "P0 final"],
    ["氟斑牙 QWK（5 类）", "0.718", "0.315", "0.143"],
    ["公开 6 类 acc", "0.820", "0.613", "0.574"],
    ["pair cosine", "0.83", "0.039", "0.030"],
    ["grayscale 余弦", "0.959", "0.301", "0.262"],
]
tbl = table(s, 0.7, 1.55, 5.45, 2.65, data,
            col_widths=[1.9, 1.05, 1.2, 1.2], font_size=12,
            highlight_rows={1: RGBColor(0xE8, 0xF2, 0xF0)})
tx(s, 0.7, 4.45, 5.45, 0.55,
   "P0 = 42k、batch 8、512、KoLeo 0.1、Sinkhorn；历史 5 类含 doubtful。",
   size=11.5, color=GRAY)
bullets(s, 6.5, 1.55, 5.95, 4.9, [
    "目标错配：DINO 优化裁剪一致性，不是类别边界。",
    "KoLeo 0.1 过强：特征近正交，pair cosine 仅 0.03。",
    "不稳定：灰度/颜色/模糊后余弦掉到 0.26–0.30。",
    "过训练：最佳 checkpoint 在 5k–10k，final 更差。",
    "窄域小数据全参数继续预训练 → 灾难性遗忘。",
], size=12.5)
tx(s, 6.5, 6.25, 5.95, 0.55,
   "教训：用下游 probe 早停，不用 training loss 选模型。",
   size=12.5, bold=True, color=NAVY)
footer(s, 15)

# ================================================================ Slide 16
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
footer(s, 16)

# ================================================================ Slide 17
s = blank()
section(s, "02 / 问题与风险")
title(s, "当前问题与风险")
data = [
    ["问题", "现状", "影响", "处理"],
    ["probe 负结果", "5k/10k/15k/16k 均低于 base", "backbone 暂不可直接用", "转配方（蒸馏/降 K）或 LoRA/全量"],
    ["SIGKILL", "16:21:41，exitcode -9", "训练中断", "已续训完成；保留 15999"],
    ["GPU 争用", "探针 1.31 → 实际 2.6 s/it", "ETA 翻倍", "协调 GPU；监控 nvidia-smi"],
    ["K/N = 0.77", "32768 prototypes / 42,616 图", "头部记忆风险", "已评 5k/10k；优先降 K"],
    ["数据重复/低分辨率", "12,941 重复；97.3% 短边 <384", "epoch 偏移、上采样", "去重视图；224 主训练 + 512 短适应"],
    ["DCP → HF", "已加载 DCP 做 probe", "下游评估", "脚本已补；继续评估 5k/10k"],
    ["数据合规", "多源 license 不同", "发表/开源风险", "逐数据集核对 license"],
]
table(s, 0.7, 1.55, 11.75, 4.9, data,
      col_widths=[1.55, 3.2, 2.2, 4.8], font_size=11.5,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])
footer(s, 17)

# ================================================================ Slide 18
s = blank()
section(s, "03 / 下一步")
title(s, "下一步计划")
data = [
    ["时间", "任务", "交付物"],
    ["本周", "5k/10k 已评估：均低于 base", "结论已明确"],
    ["本周", "base 蒸馏/冻结 Gram、降 K/KoLeo、短 schedule", "新配方"],
    ["下周", "LoRA/全量微调（氟斑牙 + 口腔疾病）", "下游适配表"],
    ["下周", "512 特征 / Stage 2（按需）", "对比结果"],
    ["2–4 周", "检测 A3（仅在 backbone 有效时）", "mAP 对比"],
    ["4 周内", "消融：K / batch / KoLeo / 配方", "消融表"],
    ["1–2 月", "论文初稿（按真实结果调整故事）", "完整 draft"],
]
table(s, 0.7, 1.55, 11.75, 4.55, data,
      col_widths=[1.1, 6.0, 4.65], font_size=12,
      aligns=[PP_ALIGN.CENTER, PP_ALIGN.LEFT, PP_ALIGN.LEFT])
shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(6.25),
                         Inches(11.75), Inches(0.6))
shp.fill.solid()
shp.fill.fore_color.rgb = RGBColor(0xFF, 0xF1, 0xE2)
shp.line.color.rgb = ORANGE
shp.shadow.inherit = False
tx(s, 0.95, 6.38, 11.3, 0.4,
   "关键 gate：5k/10k/15k/16k 都已低于 base；暂停下游大投入，转配方或 LoRA/全量。",
   size=12.5, bold=True, color=NAVY)
footer(s, 18)

# ================================================================ Slide 19
s = blank()
section(s, "03 / 论文安排")
title(s, "论文撰写安排")
tx(s, 0.7, 1.55, 6.0, 0.45,
   "暂定题目：OralDINO — Domain-Adaptive DINOv3 Pretraining for Intraoral Images",
   size=13, bold=True, color=NAVY)
bullets(s, 0.7, 2.15, 6.05, 3.9, [
    "Introduction：域鸿沟、数据分散、标注瓶颈。",
    "Related Work：DINOv3、牙科基础模型、牙齿检测。",
    "Method：数据审计与去重、continue-pretraining、小 batch softmax centering、下游适配。",
    "Experiments：probe（已完成）、LoRA/全量、检测 A3、消融。",
    "Discussion：P0 负结果与适用边界。",
    "Conclusion：可复用的口内视觉 backbone。",
], size=13)
card(s, 7.1, 1.6, 5.35, 4.55, "时间线与目标",
     "11 月：预训练 + probe（已完成）\n"
     "12 月：下游适配 + 检测 A3（下游验证）+ 消融\n"
     "1 月：论文初稿\n"
     "2 月：投稿\n\n"
     "目标会议 / 期刊：待定\n"
     "（MICCAI / ISBI / MedIA 为候选）",
     accent=TEAL, title_size=15, body_size=12.5)
tx(s, 0.7, 6.3, 11.75, 0.45,
   "原则：先把“数据—预训练—下游”闭环做扎实，再决定是否扩展新的下游任务。",
   size=12.5, bold=True, color=NAVY)
footer(s, 19)

# ================================================================ Slide 20
s = blank()
section(s, "03 / 论文安排")
title(s, "论文摘要（草稿）")
tx(s, 0.7, 1.18, 11.75, 0.32,
   "OralDINO: Domain-Adaptive DINOv3 Pretraining for Intraoral Images",
   size=12, bold=True, color=GRAY)
card(s, 0.7, 1.55, 11.75, 4.95, "摘要",
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
     accent=TEAL, title_size=15, body_size=13.5)
tx(s, 0.7, 6.62, 11.75, 0.3,
   "关键词：口腔影像 · DINOv3 · 自监督预训练 · 域自适应 · 氟斑牙分级 · 牙齿检测",
   size=10.5, color=GRAY)
footer(s, 20)

# ================================================================ Slide 21
s = blank()
section(s, "总结")
title(s, "总结与讨论")
tx(s, 0.7, 1.5, 5.9, 0.35, "三点结论", size=15, bold=True, color=TEAL)
bullets(s, 0.7, 1.95, 5.9, 3.1, [
    "检测基线已收口：8 模型 + 828 固定 test + 人工复核。",
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
footer(s, 21)

prs.save(OUT)
print(f"saved: {OUT}")
print(f"slides: {len(prs.slides)}")
