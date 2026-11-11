#!/usr/bin/env python3
"""6-class oral-disease frozen probe for the official DINOv3 Stage-1 checkpoint.

Reuses the model/feature/probe helpers from ``eval_stage1_checkpoint.py`` and
evaluates base DINOv3 vs the trained teacher/student on the 720-image balanced
Oral-Diseases subset (120 images × 6 classes).

Usage (from the project root):

    PYTHONPATH=lab_train/third_party/dinov3 \
    /tmp/dino-eval/bin/python lab_train/pretrain/eval_stage1_disease.py \
        --config lab_train/output/official-dental-vitb-224/config.yaml \
        --checkpoint lab_train/output/official-dental-vitb-224/ckpt/15999 \
        --base-checkpoint lab_train/weights/dinov3-vitb16-official.pth \
        --data-root lab_train/output/oral-diseases-720 \
        --output lab_train/output/eval-15999-disease
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_stage1_checkpoint import (  # noqa: E402
    build_model,
    collect_items,
    extract_features,
    load_base_weights,
    load_dcp,
    probe,
    summarize,
)

CLASSES = {
    "Calculus": 0,
    "Caries": 1,
    "Gingivitis": 2,
    "Hypodontia": 3,
    "Tooth Discoloration": 4,
    "Ulcers": 5,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.output.mkdir(parents=True, exist_ok=True)

    items = collect_items(args.data_root)
    paths, labels = [], []
    for path, cls in items:
        if cls not in CLASSES:
            continue
        paths.append(path)
        labels.append(CLASSES[cls])
    labels = np.asarray(labels)
    print(f"[data] {len(paths)} images, classes={sorted(CLASSES)}")

    model = build_model(args.config)
    load_base_weights(model, args.base_checkpoint)

    features = {}
    features[("base", "6-disease")] = extract_features(
        model.teacher.backbone, paths, args.imgsz, args.batch_size, "cpu"
    )
    print("[features] base done", features[("base", "6-disease")].shape)

    load_dcp(model, args.checkpoint)
    model.teacher.backbone.eval()
    model.student.backbone.eval()
    for name, backbone in (("teacher", model.teacher.backbone), ("student", model.student.backbone)):
        features[(name, "6-disease")] = extract_features(
            backbone, paths, args.imgsz, args.batch_size, "cpu"
        )
        print(f"[features] {name} done")

    summary_rows = []
    detail_rows = []
    for (model_name, protocol), X in features.items():
        rows = probe(X, labels, args.folds, args.seeds)
        s = summarize(rows)
        summary_rows.append(
            {"model": model_name, "protocol": protocol, "n_images": int(len(labels)), **s}
        )
        for r in rows:
            detail_rows.append({"model": model_name, "protocol": protocol, **r})

    with (args.output / "probe_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output / "probe_details.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    np.savez_compressed(
        args.output / "features.npz",
        **{f"{m}__{p}": X for (m, p), X in features.items()},
        labels=labels,
    )

    lines = ["# Frozen linear probe: 6-class oral disease (720 images)", ""]
    lines.append("| model | n | accuracy | macro-F1 | QWK |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for r in summary_rows:
        lines.append(
            f"| {r['model']} | {r['n_images']} | "
            f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f} | "
            f"{r['macro_f1_mean']:.3f} ± {r['macro_f1_std']:.3f} | "
            f"{r['qwk_mean']:.3f} ± {r['qwk_std']:.3f} |"
        )
    lines += [
        "",
        "Classes: Calculus / Caries / Gingivitis / Hypodontia / Tooth Discoloration / Ulcers.",
        "Features: CLS token, 224 input, ImageNet normalization.",
        "Classifier: StandardScaler + LogisticRegression(C=0.1, class_weight='balanced').",
        f"{args.folds}-fold stratified CV × {args.seeds} seeds.",
        "",
    ]
    (args.output / "probe_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
