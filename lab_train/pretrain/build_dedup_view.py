#!/usr/bin/env python3
"""Build a SHA256-deduplicated image view from ``audit_pretrain_data.py`` output.

The audit CSV has one row per scanned file.  ``duplicate_of`` only says "this
file belongs to a duplicate group"; it is **not** a keep/drop flag.  Therefore
this script groups rows by ``sha256`` itself and keeps the first readable file
of every hash.

Usage (on the training server, from ``lab_train/``)::

    python pretrain/build_dedup_view.py \
        --data-root dataset \
        --audit output/data-audit/image_audit.csv \
        --out dataset/pretrain_dedup \
        --delete-corrupt

The resulting directory contains one hardlink (or symlink) per unique image and
preserves the original relative paths, so ``ImageFolder`` can scan it directly.
Point ``train.dataset_path`` at this directory and set
``OFFICIAL_EPOCH_LENGTH = ceil(unique_count / batch_size)``.

Deleting the corrupt source files is opt-in via ``--delete-corrupt``; the audit
lists them in ``strict_ok == "False"`` rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root that the audit paths are relative to (e.g. dataset)")
    parser.add_argument("--audit", type=Path, required=True,
                        help="image_audit.csv produced by audit_pretrain_data.py")
    parser.add_argument("--out", type=Path, required=True,
                        help="Destination directory for the deduplicated view")
    parser.add_argument("--mode", choices=("hardlink", "symlink", "copy"),
                        default="hardlink",
                        help="How to materialise the view (default: hardlink)")
    parser.add_argument("--delete-corrupt", action="store_true",
                        help="Delete source files whose strict decode failed")
    parser.add_argument("--overwrite", action="store_true",
                        help="Remove an existing --out directory first")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_root = args.out.resolve()

    if not data_root.is_dir():
        sys.exit(f"data root not found: {data_root}")
    if not args.audit.is_file():
        sys.exit(f"audit CSV not found: {args.audit}")
    if out_root == data_root:
        sys.exit("--out must be a different directory from --data-root")

    if out_root.exists():
        if not args.overwrite:
            sys.exit(f"{out_root} already exists; pass --overwrite to rebuild it")
        shutil.rmtree(out_root)

    kept: dict[str, dict[str, str]] = {}
    corrupt: list[str] = []
    total = 0
    with args.audit.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total += 1
            path = row.get("path", "")
            if not path:
                continue
            if row.get("strict_ok") != "True":
                corrupt.append(path)
                continue
            if row.get("eval_overlap"):
                # Never let a downstream evaluation image leak into pretraining.
                continue
            sha = row.get("sha256", "")
            if not sha:
                continue
            kept.setdefault(sha, row)

    if args.delete_corrupt:
        deleted = 0
        for rel in corrupt:
            path = data_root / rel
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                pass
        print(f"deleted corrupt files: {deleted}/{len(corrupt)}")

    out_root.mkdir(parents=True, exist_ok=True)
    linked = 0
    missing = 0
    for row in kept.values():
        src = data_root / row["path"]
        dst = out_root / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.is_file():
            missing += 1
            continue
        try:
            if args.mode == "hardlink":
                if dst.exists():
                    dst.unlink()
                os.link(src, dst)
            elif args.mode == "symlink":
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)
            linked += 1
        except OSError as exc:
            print(f"[warn] {src}: {exc}", file=sys.stderr)
            missing += 1

    summary = {
        "audit_rows": total,
        "unique_kept": len(kept),
        "materialised": linked,
        "corrupt": len(corrupt),
        "missing": missing,
        "mode": args.mode,
        "data_root": str(data_root),
        "out": str(out_root),
    }
    (out_root / "_dedup_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSet in the config:")
    print(f"  train.dataset_path: ImageFolder:root={out_root}")
    print("  train.OFFICIAL_EPOCH_LENGTH = ceil(%d / batch_size)" % len(kept))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
