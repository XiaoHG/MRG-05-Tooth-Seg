"""Create side-by-side visual reviews for the largest prediction differences."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass
class Comparison:
    image_name: str
    before: dict
    after: dict
    matched: list[tuple[dict, dict, float]]
    before_unmatched: list[dict]
    after_unmatched: list[dict]
    score: float


def box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def compare(image_name: str, before: dict, after: dict, iou_threshold: float = 0.5) -> Comparison:
    candidates = []
    for before_index, before_detection in enumerate(before.get("detections", [])):
        for after_index, after_detection in enumerate(after.get("detections", [])):
            iou = box_iou(before_detection["box_xyxy"], after_detection["box_xyxy"])
            if iou >= iou_threshold:
                candidates.append((iou, before_index, after_index))
    candidates.sort(reverse=True)
    used_before: set[int] = set()
    used_after: set[int] = set()
    matched = []
    for iou, before_index, after_index in candidates:
        if before_index in used_before or after_index in used_after:
            continue
        used_before.add(before_index)
        used_after.add(after_index)
        matched.append((before["detections"][before_index], after["detections"][after_index], iou))
    before_unmatched = [item for index, item in enumerate(before.get("detections", [])) if index not in used_before]
    after_unmatched = [item for index, item in enumerate(after.get("detections", [])) if index not in used_after]
    matched_difference = sum((1 - iou) + abs(first["confidence"] - second["confidence"]) for first, second, iou in matched)
    score = len(before_unmatched) + len(after_unmatched) + matched_difference
    return Comparison(image_name, before, after, matched, before_unmatched, after_unmatched, score)


def labelled_pair(before_path: Path, after_path: Path, output_path: Path, max_width: int, quality: int) -> None:
    with Image.open(before_path) as before_source, Image.open(after_path) as after_source:
        before = before_source.convert("RGB")
        after = after_source.convert("RGB")
    target_width = max_width // 2
    target_height = max(1, round(before.height * min(1, target_width / before.width)))
    before = before.resize((target_width, target_height), Image.Resampling.LANCZOS)
    after = after.resize((target_width, target_height), Image.Resampling.LANCZOS)
    header_height = 44
    canvas = Image.new("RGB", (target_width * 2, target_height + header_height), "white")
    canvas.paste(before, (0, header_height))
    canvas.paste(after, (target_width, header_height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arialbd.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, target_width, header_height), fill="#202020")
    draw.rectangle((target_width, 0, target_width * 2, header_height), fill="#202020")
    draw.text((16, 8), "BEFORE", fill="#FFFF00", font=font)
    draw.text((target_width + 16, 8), "AFTER", fill="#FFFF00", font=font)
    canvas.save(output_path, format="JPEG", quality=quality, optimize=True)


def description(comparison: Comparison) -> list[str]:
    lines = [
        f"- 差异分数：`{comparison.score:.6f}`",
        f"- 检测框数量：before `{len(comparison.before.get('detections', []))}`，after `{len(comparison.after.get('detections', []))}`",
        f"- 匹配框数量（IoU >= 0.50）：`{len(comparison.matched)}`",
        f"- 未匹配框数量：before `{len(comparison.before_unmatched)}`，after `{len(comparison.after_unmatched)}`",
    ]
    if comparison.matched:
        mean_iou = sum(item[2] for item in comparison.matched) / len(comparison.matched)
        mean_confidence_delta = sum(abs(item[0]["confidence"] - item[1]["confidence"]) for item in comparison.matched) / len(comparison.matched)
        lines.extend([f"- 匹配框平均 IoU：`{mean_iou:.6f}`", f"- 匹配框平均置信度差：`{mean_confidence_delta:.6f}`"])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=None, help="Optional maximum number of comparisons.")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()

    before_json = json.loads((args.before / "predictions.json").read_text(encoding="utf-8"))
    after_json = json.loads((args.after / "predictions.json").read_text(encoding="utf-8"))
    comparisons = []
    for image_name in sorted(set(before_json) & set(after_json)):
        if "visualization" not in before_json[image_name] or "visualization" not in after_json[image_name]:
            continue
        before_overlay = args.before / before_json[image_name]["visualization"]
        after_overlay = args.after / after_json[image_name]["visualization"]
        if before_overlay.is_file() and after_overlay.is_file():
            comparisons.append(compare(image_name, before_json[image_name], after_json[image_name]))
    comparisons.sort(key=lambda item: item.score, reverse=True)
    selected = comparisons if args.top is None else comparisons[: args.top]

    args.output.mkdir(parents=True, exist_ok=True)
    readme = [
        "# YOLO11n Before/After 预测差异对比",
        "",
        "本目录仅包含派生的对比结果；`before/` 和 `after/` 中的源文件未被修改。",
        "",
        "## 覆盖范围",
        "",
        f"- before 预测记录数：`{len(before_json)}`",
        f"- after 预测记录数：`{len(after_json)}`",
        f"- 两侧均有可视化图、可直接比较的记录数：`{len(comparisons)}`",
        f"- 输出对比图数量：`{len(selected)}`",
        "- `after` 推理未完成，缺失的图像不作为模型预测差异，而是从本次排序中排除。",
        "",
        "## 排序方法",
        "",
        "先按最高 IoU 贪心匹配检测框，仅保留 IoU >= 0.50 的匹配。差异分数等于未匹配检测框数量，加上每个匹配框对的 `(1 - IoU) + 置信度绝对差`。分数越大，表示两次预测分歧越明显。",
        "",
        "## 差异最大的图像",
        "",
    ]
    for rank, item in enumerate(selected, start=1):
        name = Path(item.image_name).stem
        pair_name = f"{rank:04d}_{name}_before_after.jpg"
        labelled_pair(
            args.before / item.before["visualization"],
            args.after / item.after["visualization"],
            args.output / pair_name,
            args.max_width,
            args.jpeg_quality,
        )
        readme.extend([f"### {rank}. `{item.image_name}`", "", f"![before and after]({pair_name})", "", *description(item), ""])
    (args.output / "README.md").write_text("\n".join(readme), encoding="utf-8")


if __name__ == "__main__":
    main()
