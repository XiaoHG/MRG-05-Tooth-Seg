"""Compose side-by-side before/after prediction overlays for visual comparison.

For each test image present in both the ``before`` and ``after`` prediction
overlay folders, this writes a single composite image with the before overlay
on the left and the after overlay on the right, each labelled in a top bar.

The originals are only ever read — nothing under ``before`` or ``after`` is
modified; all composites go into a separate ``diff`` folder.

Inputs (flat ``<stem>_overlay.png`` files, as produced by ``predict-dir``):

  <base>/before/<stem>_overlay.png
  <base>/after/<stem>_overlay.png

Output:

  <base>/diff/<stem>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

LABEL_H = 46          # top label bar height (px)
SEP = 6               # horizontal gap between the two panels (px)
BAR_COLOR = (255, 255, 255)
BAR_LINE = (200, 200, 200)
TEXT_COLOR = (0, 0, 0)
SUFFIX = "_overlay.png"


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _center_text(draw: ImageDraw.ImageDraw, text: str, cx: float, cy: float, font, fill) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w = right - left
    h = bottom - top
    draw.text((cx - w / 2.0 - left, cy - h / 2.0 - top), text, fill=fill, font=font)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "output/hard-subset/visualization")
    parser.add_argument("--before", type=Path, default=None)
    parser.add_argument("--after", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    base = args.base
    before_dir = args.before or base / "before"
    after_dir = args.after or base / "after"
    out_dir = args.output or base / "diff"

    before = {p.name[: -len(SUFFIX)]: p for p in before_dir.glob(f"*{SUFFIX}")}
    after = {p.name[: -len(SUFFIX)]: p for p in after_dir.glob(f"*{SUFFIX}")}
    stems = sorted(set(before) & set(after))
    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))

    out_dir.mkdir(parents=True, exist_ok=True)
    font = _font(30)

    written = 0
    for stem in stems:
        before_img = Image.open(before[stem]).convert("RGB")
        after_img = Image.open(after[stem]).convert("RGB")
        w, h = before_img.size
        if after_img.size != (w, h):
            after_img = after_img.resize((w, h))

        canvas = Image.new("RGB", (w * 2 + SEP, h + LABEL_H), BAR_COLOR)
        canvas.paste(before_img, (0, LABEL_H))
        canvas.paste(after_img, (w + SEP, LABEL_H))

        draw = ImageDraw.Draw(canvas)
        draw.line((0, LABEL_H - 1, canvas.width, LABEL_H - 1), fill=BAR_LINE, width=1)
        _center_text(draw, "before", w / 2.0, LABEL_H / 2.0, font, TEXT_COLOR)
        _center_text(draw, "after", w + SEP + w / 2.0, LABEL_H / 2.0, font, TEXT_COLOR)

        canvas.save(out_dir / f"{stem}.png")
        written += 1

    print(f"Wrote {written} composites to {out_dir}")
    if only_before:
        print(f"Only in before ({len(only_before)}): {only_before[:10]}")
    if only_after:
        print(f"Only in after ({len(only_after)}): {only_after[:10]}")


if __name__ == "__main__":
    main()
