# -*- coding: utf-8 -*-
"""Regenerate 5-model experiment figures for the paper."""
import zipfile
import io
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)
ROOT = os.path.abspath(os.path.join(BASE, "..", "..", ".."))

MODELS = ["YOLO11n", "YOLO11s", "YOLOv8m", "YOLO26m", "RT-DETR-L"]
ZIPS = {
    "YOLO11n": ("output/train/yolo11n/omni-coco-after-result.zip",
                "output/train/yolo11n/omni-coco-before-result.zip"),
    "YOLO11s": ("output/train/yolo11s/omni-coco-after-result.zip",
                "output/train/yolo11s/omni-coco-before-result.zip"),
    "YOLOv8m": ("output/train/yolov8m/omni-coco-yolov8m-after-result.zip",
                "output/train/yolov8m/omni-coco-yolov8m-before-result.zip"),
    "YOLO26m": ("output/train/yolo26m/omni-coco-yolo26m-after-result.zip",
                "output/train/yolo26m/omni-coco-yolo26m-before-result.zip"),
    "RT-DETR-L": ("output/train/rtdetr-l/omni-coco-rtdetr-l-after-v4-result.zip",
                  "output/train/rtdetr-l/omni-coco-rtdetr-l-before-v4-result.zip"),
}

# Best-per-metric validation values (verified against per-model README reports).
BEFORE = {
    "YOLO11n":  (0.94519, 0.94945, 0.96997, 0.61778),
    "YOLO11s":  (0.94496, 0.94838, 0.96805, 0.61062),
    "YOLOv8m":  (0.94171, 0.94320, 0.96656, 0.60833),
    "YOLO26m":  (0.94687, 0.95209, 0.96922, 0.61432),
    "RT-DETR-L": (0.93651, 0.95001, 0.96633, 0.61209),
}
AFTER = {
    "YOLO11n":  (0.96169, 0.95588, 0.98439, 0.63625),
    "YOLO11s":  (0.96256, 0.95656, 0.98305, 0.63046),
    "YOLOv8m":  (0.95810, 0.94875, 0.97765, 0.62935),
    "YOLO26m":  (0.96293, 0.95586, 0.98466, 0.63199),
    "RT-DETR-L": (0.95239, 0.95664, 0.98116, 0.63845),
}
METRICS = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]


def read_curve(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        name = [n for n in zf.namelist() if n.endswith("results.csv")][0]
        df = pd.read_csv(io.BytesIO(zf.read(name)))
        df.columns = [c.strip() for c in df.columns]
    return df


def load_curves():
    curves = {}
    for m in MODELS:
        after_zip, before_zip = ZIPS[m]
        a = read_curve(os.path.join(ROOT, after_zip))
        b = read_curve(os.path.join(ROOT, before_zip))
        col = "metrics/mAP50-95(B)"
        curves[m] = (b[col].tolist(), a[col].tolist())
    return curves


def fig_metric_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    x = np.arange(len(MODELS))
    w = 0.38
    for i, (ax, metric) in enumerate(zip(axes.ravel(), METRICS)):
        before = [BEFORE[m][i] for m in MODELS]
        after = [AFTER[m][i] for m in MODELS]
        ax.bar(x - w / 2, before, w, label="原始数据集", color="#7f9cbf")
        ax.bar(x + w / 2, after, w, label="人工复核后", color="#d98f7f")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=15, ha="right")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        if metric == "mAP@0.5:0.95":
            ax.set_ylim(0.55, 0.70)
        elif metric == "mAP@0.5":
            ax.set_ylim(0.92, 1.0)
        else:
            ax.set_ylim(0.90, 1.0)
        if i == 0:
            ax.legend(fontsize=9)
    fig.suptitle("五种模型在原始与人工复核后数据集上的指标对比", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGDIR, "metric_comparison.png"), dpi=200)
    plt.close(fig)


def fig_metric_delta():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODELS))
    w = 0.2
    for i, metric in enumerate(METRICS):
        delta = [AFTER[m][i] - BEFORE[m][i] for m in MODELS]
        ax.bar(x + (i - 1.5) * w, delta, w, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.set_ylabel("指标变化量（人工复核后 − 原始）")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, ncol=4)
    ax.set_title("人工复核后相对原始数据集的指标提升幅度", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "metric_delta.png"), dpi=200)
    plt.close(fig)


def fig_curves():
    curves = load_curves()
    n = len(MODELS)
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8), sharex=True, sharey=True)
    axes = np.array(axes).ravel()
    for i, m in enumerate(MODELS):
        ax = axes[i]
        b, a = curves[m]
        epochs = np.arange(1, len(b) + 1)
        ax.plot(epochs, b, color="#7f9cbf", linestyle="--", label="原始数据集")
        ax.plot(epochs, a, color="#d98f7f", linestyle="-", label="人工复核后")
        ax.set_title(m, fontsize=12)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9, loc="lower right")
    for j in range(n, ncols * nrows):
        axes[j].axis("off")
    for ax in axes[:n]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP@0.5:0.95")
    fig.suptitle("各模型训练过程中 mAP@0.5:0.95 变化曲线（原始 vs 人工复核后）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGDIR, "map50_95_curves.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_metric_comparison()
    fig_metric_delta()
    fig_curves()
    print("figures written to", FIGDIR)
    for f in sorted(os.listdir(FIGDIR)):
        print(" -", f)
