from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toothseg.data import validate_yolo_detection_dataset
from toothseg.pipeline import prepare_dataset, predict, train, validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toothseg")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate")
    p.add_argument("--data", type=Path, default=Path("dataset/project-1"))

    p = sub.add_parser("prepare")
    p.add_argument("--data", type=Path, default=Path("dataset/project-1"))
    p.add_argument("--output", type=Path, default=Path("dataset/project-1"))
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("train")
    p.add_argument("--data-yaml", type=Path, default=Path("dataset/project-1/data.yaml"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1024)

    p = sub.add_parser("val")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data-yaml", type=Path, default=Path("dataset/project-1/data.yaml"))

    p = sub.add_parser("predict")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
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
        path = prepare_dataset(args.data, args.output, val_ratio=args.val_ratio, seed=args.seed)
        print(path)
    elif args.cmd == "train":
        print(train(args.data_yaml, epochs=args.epochs, imgsz=args.imgsz))
    elif args.cmd == "val":
        print(validate(args.weights, args.data_yaml))
    elif args.cmd == "predict":
        print(predict(args.weights, args.image, args.output, conf=args.conf))


if __name__ == "__main__":
    main()
