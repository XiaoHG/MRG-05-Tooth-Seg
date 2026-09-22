from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toothseg.data import validate_yolo_detection_dataset
from toothseg.pipeline import (
    export_pseudo_labels,
    prepare_dataset,
    predict,
    predict_dir,
    publish_pseudo_labels,
    train,
    validate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toothseg")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate")
    p.add_argument("--data", type=Path, default=Path("dataset"))

    p = sub.add_parser("prepare")
    p.add_argument("--data", type=Path, default=Path("dataset/raw"))
    p.add_argument("--output", type=Path, default=Path("dataset"))
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fixed-val-manifest", type=Path, default=None)

    p = sub.add_parser("pseudo-label-export")
    p.add_argument("--predict", type=Path, default=Path("output/predict"))
    p.add_argument("--images", type=Path, default=Path("dataset/images/test"))
    p.add_argument("--output", type=Path, default=Path("output/pseudo-label-v6-candidates"))
    p.add_argument("--confidence", type=float, default=0.9)
    p.add_argument("--teacher-weights", type=Path, default=None)

    p = sub.add_parser("pseudo-label-publish")
    p.add_argument("--candidates", type=Path, default=Path("output/pseudo-label-v6-candidates"))
    p.add_argument("--raw", type=Path, default=Path("dataset/raw"))

    p = sub.add_parser("train")
    p.add_argument("--data-yaml", type=Path, default=Path("dataset/data.yaml"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=-1)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default=None)
    p.add_argument("--project", default="output/train")
    p.add_argument("--name", default="tooth-detect-v5")

    p = sub.add_parser("val")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data-yaml", type=Path, default=Path("dataset/data.yaml"))

    p = sub.add_parser("predict")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("output/predict"))
    p.add_argument("--conf", type=float, default=0.25)

    p = sub.add_parser("predict-dir")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("output/predict"))
    p.add_argument("--conf", type=float, default=0.25)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "validate":
        summary = validate_yolo_detection_dataset(args.data)
        print(summary)
    elif args.cmd == "prepare":
        path = prepare_dataset(
            args.data, args.output, val_ratio=args.val_ratio, seed=args.seed,
            fixed_val_manifest=args.fixed_val_manifest,
        )
        print(path)
    elif args.cmd == "pseudo-label-export":
        print(export_pseudo_labels(args.predict, args.images, args.output, args.confidence, args.teacher_weights))
    elif args.cmd == "pseudo-label-publish":
        print(publish_pseudo_labels(args.candidates, args.raw))
    elif args.cmd == "train":
        print(
            train(
                args.data_yaml,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                amp=args.amp,
                device=args.device,
                project=args.project,
                name=args.name,
            )
        )
    elif args.cmd == "val":
        print(validate(args.weights, args.data_yaml))
    elif args.cmd == "predict":
        print(predict(args.weights, args.image, args.output, conf=args.conf))
    elif args.cmd == "predict-dir":
        print(f"Predicting images in: {args.image_dir}")
        print(predict_dir(args.weights, args.image_dir, args.output, conf=args.conf))


if __name__ == "__main__":
    main()
