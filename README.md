# MRG-05-Tooth-Seg

Tooth detection workspace for oral images. Current v5 is a reproducible single-class tooth detection pipeline: YOLO11 detects tooth bounding boxes from 5-column YOLO labels. It does not yet produce pixel-level segmentation masks or FDI labels.

## Status

- Raw annotated sources are kept unchanged in `dataset/raw/`.
- `dataset/images/train`, `dataset/images/val`, `dataset/labels/train`, and `dataset/labels/val` are generated from the raw sources.
- `dataset/images/test` is reserved for user-added, unlabeled prediction images.
- Training, validation, inference, and tests are available.
- v5 dataset preparation and code tests are complete.
- v6 pseudo-label candidates can be exported from prediction JSON files for human review.
- A 1-epoch CPU smoke run completed successfully; a final 50-epoch model has not yet been accepted.

## Dataset

`dataset` is organized as follows:

```text
dataset/
  raw/
    fluorosis/
    OMNI_COCO-seg-train/
  images/
    train/
    val/
    test/        # unlabeled prediction images only
  labels/
    train/
    val/
  classes.txt
  data.yaml
  split_manifest.json
  dataset_manifest.json
```

The raw sources are immutable inputs. The `prepare` command rebuilds only train/val images and labels, preserving `dataset/raw/` and `dataset/images/test`.

Format:

- Single class: `tooth`
- YOLO detection label: `class x_center y_center width height`
- Raw annotated data is split into train/val with a fixed seed.
- `images/test` has no labels and is excluded from training and quantitative validation.

## Install

```bash
pip install -e .
pip install -e .[train]
```

## Build Dataset

Validate the generated train/val dataset:

```bash
python cli/main.py validate --data dataset
```

Rebuild train/val from every compatible source under `dataset/raw/`:

```bash
python cli/main.py prepare --data dataset/raw --output dataset --val-ratio 0.2 --seed 42
```

This creates `data.yaml`, `split_manifest.json`, and `dataset_manifest.json`. The latter records the source datasets and generated file assignments.

## Train

```bash
python cli/main.py train --data-yaml dataset/data.yaml --epochs 50 --imgsz 1024 --batch -1 --project output/train --name tooth-detect-v5
```

Outputs:

- `output/train/tooth-detect-v5/weights/best.pt`
- `output/train/tooth-detect-v5/weights/last.pt`

`--batch -1` lets Ultralytics select a batch size. Add `--device 0` to select the first CUDA device when required. Use `--no-amp` when a legacy GPU has AMP-related CUDA errors.

On Windows Conda environments, NumPy/MKL and PyTorch can load duplicate Intel OpenMP runtimes. The project sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing NumPy to allow the process to start. This is a compatibility workaround, not a performance guarantee; for production training, use an environment with a single OpenMP runtime.

For a 5 GB GPU, use a conservative configuration if automatic batching fails:

```bash
python cli/main.py train --data-yaml dataset/data.yaml --epochs 50 --imgsz 640 --batch 1 --no-amp --device 0 --project output/train --name tooth-detect-v5
```

The current validated smoke weight is `output/train/tooth-detect-v5-cpu-smoke/weights/best.pt`. It was trained for one CPU epoch only and is suitable for pipeline verification, not for performance claims.

## Validate

```bash
python cli/main.py val --weights output/train/tooth-detect-v5/weights/best.pt --data-yaml dataset/data.yaml
```

The validation metrics only cover the labeled `val` split. Unlabeled test images cannot produce mAP, precision, or recall.

## Inference

Predict one image:

```bash
python cli/main.py predict --weights output/train/tooth-detect-v5/weights/best.pt --image dataset/images/test/example.jpg --output output/predict
```

Predict a directory, including the unlabeled `images/test` directory:

```bash
python cli/main.py predict-dir --weights output/train/tooth-detect-v5/weights/best.pt --image-dir dataset/images/test --output output/predict
```

Each image is written under `output/predict/<image-name>/`:

- `<image-name>_overlay.png`: full image with detection boxes
- `<image-name>_detections.json`: detection boxes and confidence values
- `<image-name>_<index>.png`: per-tooth crop

## v6 Pseudo Labels

Export `confidence >= 0.9` detections and the complete source images into a reviewable candidate dataset:

```bash
python cli/main.py pseudo-label-export --predict output/predict --images dataset/images/test --output dataset/raw/pseudo-label-v6-candidates
```

The command writes YOLO labels, source hashes, detection provenance, and `review_status: pending` records to `notes.json`. It does not modify `dataset/raw/`. After reviewing the candidates, set selected records to `review_status: approved` and publish them:

```bash
python cli/main.py pseudo-label-publish --candidates dataset/raw/pseudo-label-v6-candidates --raw dataset/raw
```

Rebuild training data while freezing the existing manually labeled validation set. Approved pseudo labels are assigned to train only:

```bash
python cli/main.py prepare --data dataset/raw --output dataset --fixed-val-manifest dataset/dataset_manifest.json --seed 42
```

The v6 candidates are not a test set and must not be used as ground truth for precision, recall, or mAP.

## Test

```bash
python -m pytest -q
```

## Notes

- The current labels train detection boxes, not pixel-level segmentation.
- Pixel-level segmentation requires per-tooth polygon or mask labels and a segmentation model.
- The current weight is not an FDI enumeration model.
- The pipeline is a research baseline and not a clinical diagnostic system.
