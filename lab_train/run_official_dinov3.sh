#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$LAB_DIR/offline_configs/config_dental_vitb_512.yaml}"
OUTPUT="${OUTPUT:-$LAB_DIR/output/official-dental-vitb-512}"
NPROC="${NPROC:-1}"
SMOKE="${SMOKE:-0}"

export PYTHONPATH="$LAB_DIR/third_party/dinov3${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$SMOKE" == "1" ]]; then
  OUTPUT="${OUTPUT}-smoke"
fi
mkdir -p "$OUTPUT"
cd "$LAB_DIR/third_party/dinov3"

ARGS=(torchrun --nproc_per_node="$NPROC" dinov3/train/train.py
      --config-file "$CONFIG" --output-dir "$OUTPUT" --no-resume)

if [[ "$SMOKE" == "1" ]]; then
  ARGS+=(train.OFFICIAL_EPOCH_LENGTH=100 optim.epochs=1)
fi

echo "config : $CONFIG"
echo "output : $OUTPUT"
echo "nproc  : $NPROC"
echo "smoke  : $SMOKE"
"${ARGS[@]}" "$@"
