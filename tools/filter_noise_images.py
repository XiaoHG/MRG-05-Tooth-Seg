"""Screen intraoral images for noise and optionally detect teeth (paper-A A1).

v14 / A1 requires a noise-image removal rule for the multi-source public dental
dataset collection. This tool implements that rule as a two-layer filter:

  static (always on, cheap, no model):
    - unreadable / corrupt file
    - under-exposed (near-black)   -> mean gray < min_brightness
    - over-exposed (white-out)     -> highlight ratio > max_highlight_ratio
    - blurred                      -> Laplacian variance < min_sharpness

  model (optional, requires --weights; ultralytics yolo11n or better):
    - no teeth detected
    - very few teeth at low confidence
        (num_detections <= min_teeth and max_conf < min_confidence)

The rationale (draft/README.md): self-supervised pretraining tolerates hard samples
but is polluted by images with no semantic content (black/white/blur/no teeth).

Safety and output:
  - dry-run by default; nothing is moved or deleted.
  - --apply moves flagged images (and their .txt labels with --with-labels) into a
    ``<output>/noise_quarantine/`` directory (reversible), never deletes.
  - refuses to run against ``dataset/raw/`` (immutable input).

  <output>/noise_filter_report.csv   per-image features + decision + reasons
  <output>/README.md                 rule summary + counts

Model inference over large collections is slow; run the --weights pass on Kaggle/Colab.
The static pass is cheap enough to run locally. Thresholds are starting defaults and
should be calibrated on a labelled sample before a full run.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "raw"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
HIGHLIGHT_GRAY = 245  # pixels at/above this gray level count as "highlight"

# 3x3 Laplacian kernel; the variance of the filtered image is a sharpness score.
LAPLACIAN = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=0)

DEFAULT_THRESHOLDS = {
    "min_brightness": 12.0,
    "max_highlight_ratio": 0.85,
    "min_sharpness": 30.0,
    "min_teeth": 2,
    "min_confidence": 0.5,
}


def image_features(path: Path) -> dict:
    """Static per-image features: brightness, highlight ratio, sharpness (Laplacian var)."""
    gray = Image.open(path).convert("L")
    stat = ImageStat.Stat(gray)
    brightness = float(stat.mean[0])

    total = max(1, gray.width * gray.height)
    hist = gray.histogram()
    bright_pixels = sum(hist[t] for t in range(HIGHLIGHT_GRAY, 256))
    highlight_ratio = bright_pixels / total

    edges = gray.filter(LAPLACIAN)
    sharpness = float(ImageStat.Stat(edges).var[0])

    return {
        "brightness": round(brightness, 2),
        "highlight_ratio": round(highlight_ratio, 4),
        "sharpness": round(sharpness, 2),
    }


def decide(features: dict, detections: list[float] | None, thr: dict) -> tuple[str, list[str]]:
    """Apply the noise rule. ``detections=None`` disables the model-based layer.

    Returns (decision, reasons) where decision is "keep" or "remove".
    """
    reasons: list[str] = []

    if features["brightness"] < thr["min_brightness"]:
        reasons.append("under_exposed")
    if features["highlight_ratio"] > thr["max_highlight_ratio"]:
        reasons.append("over_exposed")
    if features["sharpness"] < thr["min_sharpness"]:
        reasons.append("blurred")

    if detections is not None:
        num = len(detections)
        max_conf = max(detections, default=0.0)
        if num == 0:
            reasons.append("no_teeth")
        elif num <= thr["min_teeth"] and max_conf < thr["min_confidence"]:
            reasons.append("few_low_conf_teeth")

    decision = "remove" if reasons else "keep"
    return decision, reasons


def _load_detector(weights: Path):
    from ultralytics import YOLO

    return YOLO(str(weights))


def _detect(detector, image_path: Path, conf: float) -> list[float]:
    result = detector.predict(source=str(image_path), conf=conf, verbose=False)[0]
    if getattr(result, "boxes", None) is None:
        return []
    return [float(c) for c in result.boxes.conf.tolist()]


def iter_images(images_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(
            p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _render_markdown(rows: list[dict], args, remove_reasons: dict) -> str:
    total = len(rows)
    removed = [r for r in rows if r["decision"] == "remove"]
    lines = [
        "# 噪声图剔除报告（论文 A·A1）",
        "",
        f"- 图像目录：`{args.images}`（recursive={args.recursive}）",
        f"- 模型层：{'启用（' + str(args.weights) + '）' if args.weights else '未启用（仅静态规则）'}",
        f"- 阈值：min_brightness={args.min_brightness}, max_highlight_ratio={args.max_highlight_ratio}, "
        f"min_sharpness={args.min_sharpness}, min_teeth={args.min_teeth}, min_confidence={args.min_confidence}",
        "",
        f"- 扫描图像：**{total}**",
        f"- 判定剔除：**{len(removed)}**（{len(removed) / total:.1%}）",
        f"- 判定保留：**{total - len(removed)}**",
        "",
        "## 剔除原因分布",
        "",
        "| 原因 | 图数 |",
        "| --- | --- |",
    ]
    for reason, count in sorted(remove_reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {reason} | {count} |")
    lines += [
        "",
        "> 单图可能命中多个原因。静态阈值（亮度/高光/清晰度）为**起始默认值，需在人工标注样本上校准**；",
        "> 模型层（no_teeth / few_low_conf_teeth）对应 draft/README.md 的「检测不到牙齿或仅低置信度检出 1–2 颗即剔除」。",
        "",
        f"完整逐图明细：`noise_filter_report.csv`。",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory of images to scan.")
    parser.add_argument("--output", type=Path, default=ROOT / "output/a1-data-integration")
    parser.add_argument("--weights", type=Path, default=None, help="Optional yolo11n/better weights for the model layer.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--apply", action="store_true", help="Move flagged images to a quarantine dir (dry-run if unset).")
    parser.add_argument("--with-labels", action="store_true", help="Also move matching .txt labels with --apply.")
    parser.add_argument("--min-brightness", type=float, default=DEFAULT_THRESHOLDS["min_brightness"])
    parser.add_argument("--max-highlight-ratio", type=float, default=DEFAULT_THRESHOLDS["max_highlight_ratio"])
    parser.add_argument("--min-sharpness", type=float, default=DEFAULT_THRESHOLDS["min_sharpness"])
    parser.add_argument("--min-teeth", type=int, default=DEFAULT_THRESHOLDS["min_teeth"])
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_THRESHOLDS["min_confidence"])
    args = parser.parse_args()

    images_dir = args.images.resolve()
    raw_dir = RAW_DIR.resolve()
    try:
        in_raw = images_dir == raw_dir or raw_dir in images_dir.parents
    except (OSError, ValueError):
        in_raw = False
    if in_raw:
        raise SystemExit("Refusing to run against dataset/raw/ (immutable input).")

    if not images_dir.is_dir():
        raise SystemExit(f"Image dir not found: {images_dir}")

    thr = {
        "min_brightness": args.min_brightness,
        "max_highlight_ratio": args.max_highlight_ratio,
        "min_sharpness": args.min_sharpness,
        "min_teeth": args.min_teeth,
        "min_confidence": args.min_confidence,
    }

    image_paths = iter_images(images_dir, args.recursive)
    if not image_paths:
        raise SystemExit(f"No images found in: {images_dir}")

    detector = _load_detector(args.weights) if args.weights else None
    if detector is not None:
        print(f"Loaded detector from {args.weights}. Scanning {len(image_paths)} images...")

    args.output.mkdir(parents=True, exist_ok=True)
    quarantine = args.output / "noise_quarantine"
    rows: list[dict] = []
    remove_reasons: dict[str, int] = {}

    for index, path in enumerate(image_paths, start=1):
        rel = path.relative_to(images_dir).as_posix() if args.recursive else path.name
        try:
            feats = image_features(path)
        except Exception:  # corrupt / unreadable image
            rows.append({
                "image": rel, "width": "", "height": "", "brightness": "",
                "highlight_ratio": "", "sharpness": "", "num_teeth": "", "max_conf": "",
                "decision": "remove", "reasons": "unreadable",
            })
            remove_reasons["unreadable"] = remove_reasons.get("unreadable", 0) + 1
            continue

        detections = None
        if detector is not None:
            confs = _detect(detector, path, args.conf)
            detections = confs

        decision, reasons = decide(feats, detections, thr)
        rows.append({
            "image": rel,
            "width": Image.open(path).size[0],
            "height": Image.open(path).size[1],
            "brightness": feats["brightness"],
            "highlight_ratio": feats["highlight_ratio"],
            "sharpness": feats["sharpness"],
            "num_teeth": len(detections) if detections is not None else "",
            "max_conf": round(max(detections), 4) if detections else "",
            "decision": decision,
            "reasons": ";".join(reasons),
        })
        if decision == "remove":
            for reason in reasons:
                remove_reasons[reason] = remove_reasons.get(reason, 0) + 1
            if args.apply:
                dest = quarantine / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                path.rename(dest)
                if args.with_labels and path.suffix.lower() in IMAGE_SUFFIXES:
                    label = path.with_suffix(".txt")
                    if label.is_file():
                        label.rename(quarantine / (Path(rel).with_suffix(".txt")))

        if index % 200 == 0 or index == len(image_paths):
            print(f"[{index}/{len(image_paths)}]")

    csv_path = args.output / "noise_filter_report.csv"
    fieldnames = [
        "image", "width", "height", "brightness", "highlight_ratio", "sharpness",
        "num_teeth", "max_conf", "decision", "reasons",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (args.output / "README.md").write_text(_render_markdown(rows, args, remove_reasons), encoding="utf-8")

    removed = sum(1 for r in rows if r["decision"] == "remove")
    print(f"Scanned {len(rows)} images; {removed} flagged for removal.")
    print(f"Report: {csv_path}")
    if args.apply:
        print(f"Moved flagged images to {quarantine}")
    else:
        print("Dry-run: no files moved (use --apply to quarantine flagged images).")


if __name__ == "__main__":
    main()
