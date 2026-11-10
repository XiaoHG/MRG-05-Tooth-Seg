#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
DATA_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="${2:-}"; shift 2 ;;
    --python) PY="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATA_ROOT" ]]; then
  if [[ -d "$LAB_DIR/dataset/pretrain_dedup" ]]; then
    DATA_ROOT="$LAB_DIR/dataset/pretrain_dedup"
  else
    DATA_ROOT="$LAB_DIR/dataset"
  fi
  echo "INFO: --data-root not set; defaulting to $DATA_ROOT"
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "WARNING: data root does not exist: $DATA_ROOT"
  echo "         Build it first: python pretrain/build_dedup_view.py ..."
fi

echo "lab_train : $LAB_DIR"
echo "python    : $($PY -V 2>&1)"
$PY - <<'PY'
import torch
print("torch     :", torch.__version__, "| cuda:", torch.version.cuda, "| available:", torch.cuda.is_available())
PY

echo
echo "[1/4] install offline wheels (no network, torch untouched)..."
$PY -m pip install --no-index --find-links "$LAB_DIR/offline_wheels" --no-deps \
  -r "$LAB_DIR/requirements-offline.txt"

echo
echo "[2/4] patch official DINOv3 repo..."
$PY "$LAB_DIR/pretrain/official_dinov3_patch/patch_official_dinov3.py" \
  --repo "$LAB_DIR/third_party/dinov3"

echo
echo "[3/4] prepare configs..."
mkdir -p "$LAB_DIR/offline_configs"
cp "$LAB_DIR/pretrain/official_dinov3_patch/config_dental_vitb_512.yaml" \
   "$LAB_DIR/offline_configs/config_dental_vitb_512.yaml"
cp "$LAB_DIR/pretrain/official_dinov3_patch/config_dental_vitb_224.yaml" \
   "$LAB_DIR/offline_configs/config_dental_vitb_224.yaml"
cp "$LAB_DIR/pretrain/official_dinov3_patch/config_dental_vitb_224_diag.yaml" \
   "$LAB_DIR/offline_configs/config_dental_vitb_224_diag.yaml"
$PY - "$LAB_DIR" "$DATA_ROOT" <<'PY'
import sys
from pathlib import Path

lab_dir = Path(sys.argv[1])
data_root = sys.argv[2].strip()
weight = lab_dir / "weights" / "dinov3-vitb16-official.pth"
for name in (
    "config_dental_vitb_512.yaml",
    "config_dental_vitb_224.yaml",
    "config_dental_vitb_224_diag.yaml",
):
    path = lab_dir / "offline_configs" / name
    text = path.read_text(encoding="utf-8")
    text = text.replace("/PATH/TO/weights/dinov3-vitb16-official.pth", str(weight))
    if data_root:
        text = text.replace("/PATH/TO/dental_pretrain", data_root)
    path.write_text(text, encoding="utf-8")
    print("prepared:", path)
if not data_root:
    print("WARNING: --data-root not given; edit train.dataset_path in offline_configs/*.yaml")
PY

echo
echo "[4/4] verify dinov3 import..."
PYTHONPATH="$LAB_DIR/third_party/dinov3" $PY - <<'PY'
from dinov3.data.loaders import make_dataset
print("dinov3 import ok")
PY

echo
echo "Done."
echo "Diag  : CONFIG=$LAB_DIR/offline_configs/config_dental_vitb_224_diag.yaml OUTPUT=$LAB_DIR/output/official-dental-diag-224 bash $LAB_DIR/run_official_dinov3.sh"
echo "Stage1: CONFIG=$LAB_DIR/offline_configs/config_dental_vitb_224.yaml OUTPUT=$LAB_DIR/output/official-dental-vitb-224 bash $LAB_DIR/run_official_dinov3.sh"
echo "Stage2: CONFIG=$LAB_DIR/offline_configs/config_dental_vitb_512.yaml OUTPUT=$LAB_DIR/output/official-dental-vitb-224 RESUME=1 bash $LAB_DIR/run_official_dinov3.sh"
echo "Smoke : SMOKE=1 bash $LAB_DIR/run_official_dinov3.sh"
