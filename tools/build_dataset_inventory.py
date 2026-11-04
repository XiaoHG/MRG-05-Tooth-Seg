"""Build a multi-source dental dataset inventory for paper-A A1.

v14 / A1 requires a dataset manifest + a noise-removal rule. This tool produces the
manifest half: a curated registry of candidate public dental datasets (source, license,
modality, task, annotation format, scale) merged with what is actually present under
``dataset/``, plus the gap to the ~200k-image pretraining target stated in
``draft/README.md``.

The registry is the single source of truth for "which datasets exist / where they come
from"; local counts are recomputed on every run from disk, so they never go stale.

Output:
  <output>/dataset_inventory.json   registry + local scan
  <output>/dataset_inventory.md     human-readable table + gap analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# name / aliases / source / url / license / modality / task / annotation / scale /
# local (relative to ROOT) / structure / local_status / notes
DATASETS = [
    {
        "name": "OMNI_COCO",
        "aliases": ["OMNI", "OMNI Dataset"],
        "source": "OMNI: Oral Imaging for Malocclusion Issues Assessments (arXiv:2505.15637)",
        "url": "https://arxiv.org/abs/2505.15637",
        "license": "待核实（见原始来源）",
        "modality": "自然口内 RGB 照片（5 视角）",
        "task": "牙齿检测 + 错颌分类",
        "annotation": "YOLO 框（本项目统一为单类别 tooth）",
        "scale": "384 受试者 / 4166 张",
        "local": "dataset/OMNI_COCO",
        "structure": "before-after",
        "local_status": "downloaded",
        "notes": "本项目主数据；before/after = 人工复核前后两版标签（图像同集）。",
    },
    {
        "name": "AlphaDent",
        "aliases": [],
        "source": "AlphaDent: A dataset for automated tooth pathology detection (arXiv:2507.22512)",
        "url": "https://arxiv.org/pdf/2507.22512",
        "license": "待核实（见原始来源）",
        "modality": "单反口内实拍 RGB",
        "task": "牙齿实例分割 + 9 类病变检测",
        "annotation": "COCO 格式",
        "scale": "295 患者 / 1320 张",
        "local": "",
        "structure": None,
        "local_status": "partial",
        "notes": "已并入 omni_coco-alphadent（无独立本地目录，train 由 2481 → 3933，即 +1452 张）；原始 9 类标注未保留。",
    },
    {
        "name": "omni_coco-alphadent",
        "aliases": ["OMNI_COCO + AlphaDent 合并"],
        "source": "本项目合并（个人总结 #10）",
        "url": "",
        "license": "同 OMNI_COCO / AlphaDent 原许可",
        "modality": "自然口内 RGB",
        "task": "单类别牙齿检测",
        "annotation": "YOLO 框（单类别 tooth）",
        "scale": "train 3933 / val 857 / test 828",
        "local": "dataset/omni_coco-alphadent",
        "structure": "plain-yolo",
        "local_status": "downloaded",
        "notes": "OMNI_COCO 与 AlphaDent 合并后的训练集，作为「多源整合」的中间产物。",
    },
    {
        "name": "COde-Dataset",
        "aliases": ["COde"],
        "source": "A benchmark multimodal oro-dental dataset for large vision-language models (Scientific Data 2026)",
        "url": "https://www.nature.com/articles/s41597-026-07342-9",
        "license": "待核实（见原始来源）",
        "modality": "自然口内 RGB（5 万张）+ X 光片（8000+）",
        "task": "牙齿检测 / 病灶识别 / 报告生成（多模态）",
        "annotation": "多模态标注（HuggingFace 开源）",
        "scale": "5 万口内照片 + 8000+ X 光 + 病历",
        "local": "",
        "structure": None,
        "local_status": "not_downloaded",
        "notes": "DOI 10.1038/s41597-026-07342-9；规模最大，是冲 20 万张目标的核心来源。",
    },
    {
        "name": "Oral-Diseases",
        "aliases": [],
        "source": "Deep Learning for Oral Health (arXiv:2509.23100)，Kaggle 公开数据集",
        "url": "https://arxiv.org/abs/2509.23100",
        "license": "待核实（见原始来源）",
        "modality": "口内实拍 RGB",
        "task": "7 类口腔病变分类（可作检测/分类下游）",
        "annotation": "图像级分类标签",
        "scale": "12653 张（6 单病种 + 1 混合）",
        "local": "",
        "structure": None,
        "local_status": "not_downloaded",
        "notes": "类别不平衡明显，整合时需注意类别分布。",
    },
    {
        "name": "OralSDv1",
        "aliases": ["OralSDv1-main"],
        "source": "自研口腔分割开源仓库（配套正式论文较少）",
        "url": "",
        "license": "待核实（见原始来源）",
        "modality": "口内图像",
        "task": "口腔语义/实例分割",
        "annotation": "分割掩码",
        "scale": "待核实",
        "local": "",
        "structure": None,
        "local_status": "not_downloaded",
        "notes": "以代码仓库为主，需核对数据与许可。",
    },
    {
        "name": "DentalAI",
        "aliases": [],
        "source": "Caries-DETR (arXiv:2604.23718) 评测用公开数据集",
        "url": "https://arxiv.org/abs/2604.23718",
        "license": "待核实（见原始来源）",
        "modality": "口内图像",
        "task": "龋齿检测",
        "annotation": "检测框",
        "scale": "待核实",
        "local": "",
        "structure": None,
        "local_status": "not_downloaded",
        "notes": "需从 Caries-DETR 论文确认下载地址与许可。",
    },
    {
        "name": "Mendeley-dataset-19.4.2024",
        "aliases": ["Mendeley 牙科数据集"],
        "source": "A comprehensive dental dataset of six classes (Data in Brief 2024)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11470401/",
        "license": "待核实（见原始来源）",
        "modality": "口内图像",
        "task": "6 类牙科目标检测",
        "annotation": "检测框",
        "scale": "待核实",
        "local": "",
        "structure": None,
        "local_status": "not_downloaded",
        "notes": "Mendeley 平台开源。",
    },
    {
        "name": "BMC-Oral-Health-caries",
        "aliases": ["BMC Oral Health 龋齿数据集"],
        "source": "Annotated intraoral image dataset for dental caries detection (BMC Oral Health 2025)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12297690/",
        "license": "待核实（见原始来源）",
        "modality": "青少年口内实拍 RGB",
        "task": "龋齿检测",
        "annotation": "YOLO / COCO / VOC",
        "scale": "6313 张",
        "local": "",
        "structure": None,
        "local_status": "not_downloaded",
        "notes": "PMC 开源，适合 YOLO 系对比。",
    },
    {
        "name": "TLNM",
        "aliases": [],
        "source": "Externally Validated Tooth Detection, Numbering and Segmentation (arXiv:2608.06275)",
        "url": "https://arxiv.org/abs/2608.06275",
        "license": "待核实（见原始来源）",
        "modality": "智能手机口内照片",
        "task": "牙齿检测 + FDI 编号 + 实例分割（Mask R-CNN）",
        "annotation": "多边形（FDI 牙位）",
        "scale": "1272 张",
        "local": "",
        "structure": None,
        "local_status": "reference_only",
        "notes": "作为基线/方法参考，非数据整合来源。",
    },
    {
        "name": "自建氟斑牙 (positive)",
        "aliases": [],
        "source": "本项目自建（论文 B 氟斑牙分级）",
        "url": "",
        "license": "自建（未公开）",
        "modality": "口内图像（分级图）",
        "task": "氟斑牙严重程度分级",
        "annotation": "按 normal/mild/medium/severe/doubtful 目录分级",
        "scale": "154 个文件（约 77 张图）",
        "local": "dataset/positive",
        "structure": "class-dirs",
        "local_status": "downloaded",
        "notes": "论文 B 数据，不属于「多源公开牙科检测数据」，此处仅作本地盘点。",
    },
]


def _count(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.is_file()) if d.is_dir() else 0


def _scan_before_after(root: Path) -> dict:
    detail: dict[str, dict] = {}
    for scope in ("before", "after"):
        detail[scope] = {}
        for part in ("images", "labels"):
            for split in ("train", "val", "test"):
                detail[scope][f"{part}/{split}"] = _count(root / scope / part / split)
    return detail


def _scan_plain_yolo(root: Path) -> dict:
    detail: dict[str, int] = {}
    for part in ("images", "labels"):
        for split in ("train", "val", "test"):
            detail[f"{part}/{split}"] = _count(root / part / split)
    return detail


def _scan_class_dirs(root: Path) -> dict:
    return {
        d.name: sum(1 for p in d.iterdir() if p.is_file())
        for d in sorted(root.iterdir()) if d.is_dir()
    }


def _scan_local(root: Path) -> dict:
    images = 0
    labels = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            images += 1
        elif suffix == ".txt":
            labels += 1
    return {"images": images, "labels": labels}


def _scan_dataset(entry: dict) -> dict:
    local = entry.get("local") or ""
    root = ROOT / local if local else None
    if root is None or not root.is_dir():
        return {"exists": False, "images": 0, "labels": 0}

    info = _scan_local(root)
    info["exists"] = True
    structure = entry.get("structure")
    if structure == "before-after":
        info["detail"] = _scan_before_after(root)
    elif structure == "plain-yolo":
        info["detail"] = _scan_plain_yolo(root)
    elif structure == "class-dirs":
        info["detail"] = _scan_class_dirs(root)
    else:
        info["detail"] = {}
    return info


def _entry_by_name(manifest: dict, name: str) -> dict | None:
    for entry in manifest["datasets"]:
        if entry["name"] == name:
            return entry
    return None


def _render_markdown(manifest: dict) -> str:
    local = manifest["local_summary"]
    target = manifest["target_pretraining_images"]

    omni = _entry_by_name(manifest, "OMNI_COCO")["scan"]
    merged = _entry_by_name(manifest, "omni_coco-alphadent")["scan"]
    fluoro = _entry_by_name(manifest, "自建氟斑牙 (positive)")["scan"]
    unique_estimate = merged["images"] + fluoro["images"]

    lines = [
        "# 多源牙科数据集清单（论文 A·A1）",
        "",
        "## 本地数据现状",
        "",
        f"- 本地磁盘图像文件总数（含 `OMNI_COCO` before/after 同图两版、合并集等重复）：**{local['downloaded_images']}** 张",
        f"- 本地标签文件总数：**{local['downloaded_labels']}** 个",
        f"- 去重后唯一自然口内图像约：合并集 **{merged['images']}** 张 + 氟斑牙 **{fluoro['images']}** 张 = **{unique_estimate}** 张",
        f"- 预训练目标（draft/README.md）：约 **{target}** 张",
        f"- 缺口：约 **{max(0, target - unique_estimate)}** 张（见下方缺口分析）",
        "",
        "## 数据集清单",
        "",
        "| 数据集 | 来源 | 模态 | 任务 | 标注 | 规模 | 本地状态 | 本地图像数 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in manifest["datasets"]:
        scan = entry["scan"]
        local_images = scan["images"] if scan["exists"] else "—"
        lines.append(
            f"| {entry['name']} | {entry['source']} | {entry['modality']} | {entry['task']} "
            f"| {entry['annotation']} | {entry['scale']} | {entry['local_status']} | {local_images} |"
        )
    lines += [
        "",
        "## 本地数据明细（自动扫描）",
        "",
    ]
    for entry in manifest["datasets"]:
        scan = entry["scan"]
        if not scan["exists"]:
            continue
        lines.append(f"### {entry['name']}")
        lines.append("")
        if scan.get("detail"):
            lines.append("| 位置 | 数量 |")
            lines.append("| --- | --- |")
            for key, value in _flatten_detail(scan["detail"]):
                lines.append(f"| {key} | {value} |")
            lines.append("")
        else:
            lines.append(f"- 图像 {scan['images']} / 标签 {scan['labels']}")
            lines.append("")
    lines += [
        "## 缺口分析",
        "",
        f"- 现有本地唯一自然口内图像约 **{unique_estimate}** 张（合并集 {merged['images']} + 氟斑牙 {fluoro['images']}）；`OMNI_COCO` 的 before/after 是同一图像集的两版标签，不重复计入口径。",
        "- 公开可下载规模（参考文献口径）：COde 5 万 + Oral-Diseases 1.27 万 + BMC 6313 + AlphaDent 1320 + TLNM 1272 ≈ 7 万张量级。",
        "- 因此「约 20 万张」是愿景值；现实可行目标为**数万张**量级，需在论文中如实说明整合规模，避免夸大。",
        "- 噪声图剔除（见 `tools/filter_noise_images.py`）应在下载各数据集后、合并前统一执行，避免黑屏/过曝/模糊/无牙图污染预训练特征。",
        "",
        "> 本地数量由脚本每次运行从磁盘重算；许可/规模等元数据来自 `references/README.md`，标注「待核实」的字段需在下载前核对原始来源。",
    ]
    return "\n".join(lines)


def _flatten_detail(detail: dict) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for key, value in detail.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                out.append((f"{key}/{sub_key}", sub_value))
        else:
            out.append((key, value))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/a1-data-integration")
    args = parser.parse_args()

    datasets = []
    downloaded_images = 0
    downloaded_labels = 0
    for entry in DATASETS:
        scan = _scan_dataset(entry)
        if scan["exists"]:
            downloaded_images += scan["images"]
            downloaded_labels += scan["labels"]
        datasets.append({**entry, "scan": scan})

    manifest = {
        "generated_from": "tools/build_dataset_inventory.py",
        "target_pretraining_images": 200000,
        "local_summary": {
            "downloaded_images": downloaded_images,
            "downloaded_labels": downloaded_labels,
        },
        "datasets": datasets,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "dataset_inventory.json"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = args.output / "dataset_inventory.md"
    md_path.write_text(_render_markdown(manifest), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Local: {downloaded_images} images / {downloaded_labels} labels across downloaded datasets")


if __name__ == "__main__":
    main()
