"""Audit a pretraining image directory before DINOv3 continue-pretraining.

Reports source distribution, duplicates (SHA256), corrupt/truncated images,
resolution/mode statistics, grayscale-like (possible X-ray) fractions, and
optional overlap with a downstream evaluation set.

Example (run from ``lab_train/``):

    python pretrain/audit_pretrain_data.py \
      --data-root dataset \
      --eval-root ../dataset/positive \
      --output output/data-audit

The script only reads images; it never modifies or deletes them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CHUNK = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def image_stats(path: Path) -> dict:
    row = {
        "format": "",
        "mode": "",
        "width": "",
        "height": "",
        "brightness": "",
        "saturation": "",
        "strict_ok": False,
        "tolerant_ok": False,
        "error": "",
    }
    try:
        with Image.open(path) as image:
            row["format"] = image.format or ""
            row["mode"] = image.mode
            row["width"], row["height"] = image.size
            rgb = image.convert("RGB")
            stat = ImageStat.Stat(rgb)
            row["brightness"] = round(sum(stat.mean) / 3.0, 2)
            hsv = rgb.convert("HSV")
            row["saturation"] = round(ImageStat.Stat(hsv).mean[1] / 255.0, 4)
            image.load()
            row["strict_ok"] = True
            row["tolerant_ok"] = True
    except Exception as exc:  # noqa: BLE001 - report whatever Pillow raises
        row["error"] = f"{type(exc).__name__}: {exc}"
    if not row["strict_ok"]:
        from PIL import ImageFile

        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(path) as image:
                image.load()
            row["tolerant_ok"] = True
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous
    return row


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--eval-root", type=Path, default=None,
                        help="Optional downstream eval directory to check for hash overlap.")
    parser.add_argument("--output", type=Path, default=ROOT / "output/data-audit")
    parser.add_argument("--sample", type=int, default=0,
                        help="Audit only the first N images per source (0 = all).")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Data root not found: {data_root}")

    images = list(iter_images(data_root))
    if args.sample > 0:
        by_source = defaultdict(list)
        for path in images:
            rel = path.relative_to(data_root)
            source = rel.parts[0] if len(rel.parts) > 1 else "."
            by_source[source].append(path)
        images = [p for paths in by_source.values() for p in paths[: args.sample]]
    print(f"Auditing {len(images)} images under {data_root}")

    eval_hashes = {}
    if args.eval_root:
        eval_root = args.eval_root.resolve()
        if eval_root.is_dir():
            for path in iter_images(eval_root):
                eval_hashes[sha256(path)] = str(path.relative_to(eval_root))
            print(f"Eval set: {len(eval_hashes)} images from {eval_root}")
        else:
            print(f"WARNING: eval root not found: {eval_root}")

    rows = []
    hashes = defaultdict(list)
    for index, path in enumerate(images, 1):
        rel = path.relative_to(data_root)
        source = rel.parts[0] if len(rel.parts) > 1 else "."
        digest = sha256(path)
        hashes[digest].append(str(rel))
        row = {
            "path": str(rel),
            "source": source,
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
            **image_stats(path),
        }
        row["eval_overlap"] = eval_hashes.get(digest, "")
        row["duplicate_of"] = ""
        rows.append(row)
        if index % 1000 == 0:
            print(f"  {index}/{len(images)}", flush=True)

    duplicate_groups = {digest: paths for digest, paths in hashes.items() if len(paths) > 1}
    for row in rows:
        group = duplicate_groups.get(row["sha256"])
        if group:
            row["duplicate_of"] = "|".join(p for p in group if p != row["path"])

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "image_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    source_counts = Counter(row["source"] for row in rows)
    strict_fail = [row for row in rows if not row["strict_ok"]]
    tolerant_fail = [row for row in rows if not row["tolerant_ok"]]
    zero = [row for row in rows if row["size_bytes"] == 0]
    duplicates = sum(len(v) - 1 for v in duplicate_groups.values())
    grayscale_like = [row for row in rows if str(row["mode"]) in {"L", "1"} or (
        isinstance(row["saturation"], float) and row["saturation"] < 0.05
    )]
    eval_overlap = [row for row in rows if row["eval_overlap"]]
    widths = [row["width"] for row in rows if isinstance(row["width"], int)]
    heights = [row["height"] for row in rows if isinstance(row["height"], int)]

    summary = {
        "data_root": str(data_root),
        "images": len(rows),
        "sources": dict(source_counts),
        "unique_sha256": len(hashes),
        "duplicate_files": duplicates,
        "duplicate_groups": len(duplicate_groups),
        "zero_byte": len(zero),
        "strict_decode_fail": len(strict_fail),
        "tolerant_decode_fail": len(tolerant_fail),
        "grayscale_like": len(grayscale_like),
        "grayscale_like_fraction": round(len(grayscale_like) / max(1, len(rows)), 4),
        "eval_overlap": len(eval_overlap),
        "width_min": min(widths) if widths else None,
        "width_max": max(widths) if widths else None,
        "height_min": min(heights) if heights else None,
        "height_max": max(heights) if heights else None,
        "width_median": statistics.median(widths) if widths else None,
        "height_median": statistics.median(heights) if heights else None,
        "csv": str(csv_path),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 预训练数据审计",
        "",
        f"- 数据根目录：`{data_root}`",
        f"- 图像数：**{len(rows)}**",
        f"- 唯一 SHA256：**{len(hashes)}**；重复文件：{duplicates}（{len(duplicate_groups)} 组）",
        f"- 零字节：{len(zero)}",
        f"- 严格解码失败：{len(strict_fail)}；容错也不可读：{len(tolerant_fail)}",
        f"- 灰度/低饱和（疑似 X 光）：{len(grayscale_like)}（{summary['grayscale_like_fraction']:.1%}）",
        f"- 与评测集哈希重叠：{len(eval_overlap)}",
        "",
        "## 来源分布",
        "",
        "| 来源 | 图像数 |",
        "| --- | ---: |",
    ]
    lines += [f"| {source} | {count} |" for source, count in source_counts.most_common()]
    if strict_fail:
        lines += ["", "## 严格解码失败（前 50）", ""]
        lines += [
            f"- `{row['path']}` — {row['error']}" for row in strict_fail[:50]
        ]
    if eval_overlap:
        lines += ["", "## 与评测集重叠（前 50）", ""]
        lines += [
            f"- `{row['path']}` ↔ `{row['eval_overlap']}`" for row in eval_overlap[:50]
        ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Report: {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
