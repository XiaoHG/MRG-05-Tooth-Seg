# MRG-05-Tooth-Seg

Tooth detection and segmentation workspace for oral images.  
Current v1 is a minimal closed loop for **single-class tooth detection**.

## Status

- Dataset is prepared in `dataset/project-1`
- Data is split into train/val in place
- Training, validation, inference, and tests are wired
- Current labels are YOLO detection boxes, not segmentation masks

## Dataset

`dataset/project-1` contains:

- `images/train`, `images/val`
- `labels/train`, `labels/val`
- `raw/images`, `raw/labels` for backup
- `classes.txt`
- `notes.json`
- `data.yaml`

Format:

- Single class: `tooth`
- YOLO label format: `class x_center y_center width height`
- Image sizes may vary

## Install

```bash
pip install -e .
pip install -e .[train]
```

## Validate Dataset

```bash
python cli/main.py validate --data dataset/project-1
```

## Prepare Dataset

Re-split raw data in place:

```bash
python cli/main.py prepare --data dataset/project-1 --output dataset/project-1
```

## Train

```bash
python cli/main.py train --data-yaml dataset/project-1/data.yaml --epochs 50 --imgsz 1024
```

Outputs:

- `runs/detect/output/train/tooth-detect/weights/best.pt`
- `runs/detect/output/train/tooth-detect/weights/last.pt`

## Validate

```bash
python cli/main.py val --weights runs/detect/output/train/tooth-detect/weights/best.pt --data-yaml dataset/project-1/data.yaml
```

Outputs:

- `runs/detect/output/val/tooth-detect`

## Inference

```bash
python cli/main.py predict --weights runs/detect/output/train/tooth-detect/weights/best.pt --image dataset/project-1/images/train/57f1bdd8-33.jpg --output output/predict
```

Outputs:

- `output/predict/<image-name>/<image-name>_overlay.png`
- `output/predict/<image-name>/<image-name>_detections.json`
- `output/predict/<image-name>/<image-name>_1.png`
- `output/predict/<image-name>/<image-name>_2.png`

## Test

```bash
python -m pytest -q
```

## Notes

- `YOLO11` is used for the minimal detection loop
- Segmentation will require mask-style labels later
- The current code is optimized for a fast baseline, not a full clinical-grade pipeline
