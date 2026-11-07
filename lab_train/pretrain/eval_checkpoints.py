"""Evaluate saved pretraining checkpoints with a frozen linear probe.

This is the P0 "periodic downstream probe": run it on ``ckpt_step*.pt`` and
``teacher_checkpoint.pt`` to choose a checkpoint by downstream QWK/F1 instead
of by training loss. The default dataset is the local 5-grade fluorosis set
(``normal/doubtful/mild/medium/severe``).

Example (run from ``lab_train/``):

    python pretrain/eval_checkpoints.py \
      --run-dir output/pretrain/oraldino-vitb16-512-42k-p0 \
      --data-root ../dataset/positive \
      --imgsz 512 --device cuda:0

Add ``--student`` to evaluate ``student_backbone`` (present when training used
``--save-student``). Results are written to ``<run-dir>/probe_results.csv``.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]  # lab_train package root
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="weights/dinov3-vitb16")
    parser.add_argument("--data-root", type=Path, default=ROOT / "../dataset/positive")
    parser.add_argument(
        "--labels",
        default="normal,doubtful,mild,medium,severe",
        help="Comma-separated ordered class folders; use 'auto' for sorted subdirs.",
    )
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--student", action="store_true",
                        help="Evaluate student_backbone instead of teacher backbone.")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name.startswith("cuda") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_images(data_root: Path, labels_arg: str):
    if labels_arg.strip().lower() == "auto":
        labels = sorted(
            p.name for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
    else:
        labels = [x.strip() for x in labels_arg.split(",") if x.strip()]
    paths, y = [], []
    for index, label in enumerate(labels):
        folder = data_root / label
        if not folder.is_dir():
            raise SystemExit(f"Missing class folder: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path)
                y.append(index)
    if not paths:
        raise SystemExit(f"No images found under {data_root} (labels={labels})")
    return paths, np.asarray(y), labels


def extract_features(model, paths, imgsz: int, batch: int, device: torch.device):
    tf = transforms.Compose([
        transforms.Resize(imgsz),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    feats = []
    with torch.no_grad():
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            tensor = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(device)
            feats.append(model(pixel_values=tensor).last_hidden_state[:, 0].cpu())
    return torch.cat(feats, dim=0).numpy()


def linear_probe(x, y, fold_count: int, repeats: int, c_value: float):
    n_classes = int(y.max()) + 1
    cv = RepeatedStratifiedKFold(n_splits=fold_count, n_repeats=repeats, random_state=0)
    accs, f1s, qwks = [], [], []
    for train, test in cv.split(x, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=c_value),
        )
        clf.fit(x[train], y[train])
        pred = clf.predict(x[test])
        accs.append(accuracy_score(y[test], pred))
        f1s.append(f1_score(y[test], pred, average="macro"))
        qwks.append(
            cohen_kappa_score(
                y[test], pred, weights="quadratic", labels=list(range(n_classes))
            )
        )
    return (
        float(np.mean(accs)), float(np.std(accs)),
        float(np.mean(f1s)), float(np.std(f1s)),
        float(np.mean(qwks)), float(np.std(qwks)),
    )


def find_checkpoints(run_dir: Path):
    items = []
    for path in run_dir.glob("ckpt_step*.pt"):
        match = re.search(r"ckpt_step(\d+)", path.name)
        if match:
            items.append((int(match.group(1)), path))
    final = run_dir / "teacher_checkpoint.pt"
    if final.exists():
        items.append((10 ** 12, final))
    items.sort()
    return items


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    paths, y, labels = collect_images(args.data_root, args.labels)
    print(f"Probe data: {len(paths)} images, labels={labels}, device={device}")

    rows = []
    for sort_step, ckpt_path in find_checkpoints(args.run_dir):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        key = "student_backbone" if args.student else "backbone"
        if key not in ckpt:
            print(f"SKIP {ckpt_path.name}: no '{key}' (train with --save-student)")
            continue
        model = AutoModel.from_pretrained(args.model)
        missing, unexpected = model.load_state_dict(ckpt[key], strict=False)
        if missing or unexpected:
            print(f"WARN {ckpt_path.name}: missing={len(missing)} unexpected={len(unexpected)}")
        model.to(device).eval()
        features = extract_features(model, paths, args.imgsz, args.batch, device)
        acc, acc_std, f1, f1_std, qwk, qwk_std = linear_probe(
            features, y, args.folds, args.repeats, args.C
        )
        step = int(ckpt.get("step", sort_step))
        row = {
            "checkpoint": ckpt_path.name,
            "step": step,
            "weights": key,
            "accuracy": round(acc, 4),
            "accuracy_std": round(acc_std, 4),
            "macro_f1": round(f1, 4),
            "macro_f1_std": round(f1_std, 4),
            "qwk": round(qwk, 4),
            "qwk_std": round(qwk_std, 4),
            "n_images": len(paths),
        }
        rows.append(row)
        print(
            f"{ckpt_path.name:<28} step={step:<7} acc={acc:.4f}±{acc_std:.4f} "
            f"f1={f1:.4f}±{f1_std:.4f} qwk={qwk:.4f}±{qwk_std:.4f}"
        )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    out = args.out or (args.run_dir / "probe_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {out}")

    if rows:
        best = max(rows, key=lambda row: (row["qwk"], row["macro_f1"]))
        print(
            f"Best by QWK: {best['checkpoint']} (step {best['step']}, "
            f"qwk={best['qwk']:.4f}, f1={best['macro_f1']:.4f})"
        )


if __name__ == "__main__":
    main()
