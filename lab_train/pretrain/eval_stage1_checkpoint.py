#!/usr/bin/env python3
"""Frozen linear probe for the official DINOv3 Stage-1 checkpoint.

Loads the DCP checkpoint (e.g. ``ckpt/15999``) into the official SSLMetaArch,
extracts CLS features for the fluorosis images, and runs a repeated
stratified 5-fold logistic-regression probe.  The base DINOv3 official
checkpoint is evaluated with the same protocol for a fair comparison.

Usage (from the project root):

    PYTHONPATH=lab_train/third_party/dinov3 \
    /tmp/dino-eval/bin/python lab_train/pretrain/eval_stage1_checkpoint.py \
        --config lab_train/output/official-dental-vitb-224/config.yaml \
        --checkpoint lab_train/output/official-dental-vitb-224/ckpt/15999 \
        --base-checkpoint lab_train/weights/dinov3-vitb16-official.pth \
        --data-root dataset/positive \
        --output lab_train/output/eval-15999

The fluorosis task is defined as 4 levels (normal / mild / moderate /
severe).  The current folder still has ``doubtful`` and ``medium``; the
script evaluates two mappings:

* ``4-exclude``: drop ``doubtful``; ``medium`` -> moderate.
* ``4-merge``:   merge ``doubtful`` into mild; ``medium`` -> moderate.

It also reports the historical 5-class protocol for reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Make the vendored repo importable when the caller did not set PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "dinov3"))

from dinov3.checkpointer import load_checkpoint  # noqa: E402
from dinov3.train.ssl_meta_arch import SSLMetaArch  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=int, default=5)
    return p.parse_args()


def build_model(cfg_path: Path) -> SSLMetaArch:
    cfg = OmegaConf.load(cfg_path)
    model = SSLMetaArch(cfg).to_empty(device="cpu")
    model.teacher.backbone.eval()
    model.student.backbone.eval()
    return model


def load_base_weights(model: SSLMetaArch, base_path: Path) -> None:
    ckpt = torch.load(base_path, map_location="cpu")
    sd = ckpt["teacher"] if isinstance(ckpt, dict) and "teacher" in ckpt else ckpt
    sd = {k.replace("backbone.", "", 1): v for k, v in sd.items()}
    msg = model.teacher.backbone.load_state_dict(sd, strict=False)
    print(f"[base] loaded {base_path} missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")


def load_dcp(model: SSLMetaArch, ckpt_path: Path) -> int:
    dist.init_process_group(
        backend="gloo", init_method="tcp://127.0.0.1:29511", rank=0, world_size=1
    )
    iteration = load_checkpoint(ckpt_path, model=model, strict_loading=False)
    print(f"[trained] loaded {ckpt_path} at iteration {iteration}")
    return iteration


def collect_items(data_root: Path):
    items = []
    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        label = class_dir.name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith("._"):
                items.append((path, label))
    return items


def protocol_labels(items, protocol: str):
    paths, labels, names = [], [], []
    for path, cls in items:
        if protocol == "4-exclude":
            if cls == "doubtful":
                continue
            mapping = {"normal": 0, "mild": 1, "medium": 2, "severe": 3}
            if cls not in mapping:
                continue
            y = mapping[cls]
        elif protocol == "4-merge":
            mapping = {"normal": 0, "doubtful": 1, "mild": 1, "medium": 2, "severe": 3}
            if cls not in mapping:
                continue
            y = mapping[cls]
        elif protocol == "5-class":
            mapping = {"normal": 0, "doubtful": 1, "mild": 2, "medium": 3, "severe": 4}
            if cls not in mapping:
                continue
            y = mapping[cls]
        else:
            raise ValueError(protocol)
        paths.append(path)
        labels.append(y)
        names.append(cls)
    return paths, np.asarray(labels), names


def make_transform(imgsz: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(int(imgsz * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


@torch.no_grad()
def extract_features(backbone, paths, imgsz: int, batch_size: int, device: str):
    tf = make_transform(imgsz)
    feats = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        tensors = [tf(Image.open(p).convert("RGB")) for p in batch]
        x = torch.stack(tensors).to(device)
        out = backbone.forward_features(x)
        feats.append(out["x_norm_clstoken"].float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def probe(X, y, folds: int, seeds: int):
    rows = []
    for seed in range(seeds):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"),
            )
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "n_test": int(len(te)),
                    "accuracy": float(accuracy_score(y[te], pred)),
                    "macro_f1": float(f1_score(y[te], pred, average="macro")),
                    "qwk": float(cohen_kappa_score(y[te], pred, weights="quadratic")),
                }
            )
    return rows


def summarize(rows):
    out = {}
    for key in ("accuracy", "macro_f1", "qwk"):
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std(ddof=1))
    out["n_folds"] = len(rows)
    return out


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.output.mkdir(parents=True, exist_ok=True)

    items = collect_items(args.data_root)
    print(f"[data] {len(items)} images under {args.data_root}")

    model = build_model(args.config)
    load_base_weights(model, args.base_checkpoint)

    all_features = {}
    all_labels = {}
    all_paths = {}
    for protocol in ("4-exclude", "4-merge", "5-class"):
        paths, labels, names = protocol_labels(items, protocol)
        all_paths[protocol] = paths
        all_labels[protocol] = labels
        print(f"[protocol] {protocol}: {len(paths)} images, classes={sorted(set(labels.tolist()))}")

    # base features (teacher == student at initialization)
    for protocol, paths in all_paths.items():
        all_features[("base", protocol)] = extract_features(
            model.teacher.backbone, paths, args.imgsz, args.batch_size, args.device
        )
        print(f"[features] base/{protocol}: {all_features[('base', protocol)].shape}")

    # trained DCP
    load_dcp(model, args.checkpoint)
    model.teacher.backbone.eval()
    model.student.backbone.eval()
    for protocol, paths in all_paths.items():
        all_features[("teacher", protocol)] = extract_features(
            model.teacher.backbone, paths, args.imgsz, args.batch_size, args.device
        )
        all_features[("student", protocol)] = extract_features(
            model.student.backbone, paths, args.imgsz, args.batch_size, args.device
        )
        print(f"[features] teacher/student {protocol}: done")

    summary_rows = []
    detail_rows = []
    for (model_name, protocol), X in all_features.items():
        y = all_labels[protocol]
        rows = probe(X, y, args.folds, args.seeds)
        s = summarize(rows)
        summary_rows.append(
            {
                "model": model_name,
                "protocol": protocol,
                "n_images": int(len(y)),
                **s,
            }
        )
        for r in rows:
            detail_rows.append({"model": model_name, "protocol": protocol, **r})

    import csv

    summary_path = args.output / "probe_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    detail_path = args.output / "probe_details.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    np.savez_compressed(
        args.output / "features.npz",
        **{
            f"{m}__{p}": X
            for (m, p), X in all_features.items()
        },
        labels_4exclude=all_labels["4-exclude"],
        labels_4merge=all_labels["4-merge"],
        labels_5class=all_labels["5-class"],
    )

    lines = ["# Frozen linear probe: checkpoint 15999", ""]
    lines.append("| model | protocol | n | accuracy | macro-F1 | QWK |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for r in summary_rows:
        lines.append(
            f"| {r['model']} | {r['protocol']} | {r['n_images']} | "
            f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f} | "
            f"{r['macro_f1_mean']:.3f} ± {r['macro_f1_std']:.3f} | "
            f"{r['qwk_mean']:.3f} ± {r['qwk_std']:.3f} |"
        )
    lines += [
        "",
        "Notes:",
        "- `4-exclude`: drop doubtful; medium -> moderate.",
        "- `4-merge`: merge doubtful into mild; medium -> moderate.",
        "- `5-class`: historical protocol with doubtful as its own class.",
        "- Features: CLS token, 224 input, ImageNet normalization.",
        "- Classifier: StandardScaler + LogisticRegression(C=0.1, class_weight='balanced').",
        f"- {args.folds}-fold stratified CV × {args.seeds} seeds.",
        "",
    ]
    (args.output / "probe_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
