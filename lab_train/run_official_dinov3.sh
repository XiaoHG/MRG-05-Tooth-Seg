#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$LAB_DIR/offline_configs/config_dental_vitb_224.yaml}"
OUTPUT="${OUTPUT:-$LAB_DIR/output/official-dental-vitb-224}"
NPROC="${NPROC:-1}"
SMOKE="${SMOKE:-0}"
RESUME="${RESUME:-0}"
ALLOW_SINKHORN="${ALLOW_SINKHORN:-0}"

export PYTHONPATH="$LAB_DIR/third_party/dinov3${PYTHONPATH:+:$PYTHONPATH}"

# If setup_offline_dinov3.sh was not re-run after copying new patch files, a
# generated config may be missing.  Regenerate it from the source config so the
# run does not fail later inside train.py with FileNotFoundError.
prepare_config() {
  local base src data_root
  base="$(basename "$CONFIG")"
  src="$LAB_DIR/pretrain/official_dinov3_patch/$base"
  [[ -f "$src" ]] || return 1
  data_root="$LAB_DIR/dataset/pretrain_dedup"
  [[ -d "$data_root" ]] || data_root="$LAB_DIR/dataset"
  mkdir -p "$(dirname "$CONFIG")"
  sed \
    -e "s#/PATH/TO/weights/dinov3-vitb16-official.pth#$LAB_DIR/weights/dinov3-vitb16-official.pth#" \
    -e "s#/PATH/TO/dental_pretrain#$data_root#" \
    "$src" > "$CONFIG"
  echo "WARNING: regenerated $CONFIG from $src" >&2
  echo "         data_root=$data_root" >&2
  echo "         Recommended: re-run setup_offline_dinov3.sh to refresh all configs." >&2
}

if [[ ! -f "$CONFIG" ]]; then
  if ! prepare_config; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "Run: bash $LAB_DIR/setup_offline_dinov3.sh --data-root $LAB_DIR/dataset/pretrain_dedup" >&2
    exit 1
  fi
fi

# A config copied by hand from the patch dir still has /PATH/TO/... and will
# only fail later inside train.py.  Regenerate it now.
if grep -Eq '^[[:space:]]*(dataset_path|resume_from_teacher_chkpt):.*(/PATH/TO/)' "$CONFIG"; then
  echo "WARNING: $CONFIG still contains /PATH/TO placeholders." >&2
  if ! prepare_config; then
    echo "ERROR: could not regenerate $CONFIG; run setup_offline_dinov3.sh." >&2
    exit 1
  fi
fi

# setup_offline_dinov3.sh always overwrites offline_configs from the patch dir.
# A newer source config means setup was not re-run after the last rsync.
SRC_CONFIG="$LAB_DIR/pretrain/official_dinov3_patch/$(basename "$CONFIG")"
if [[ -f "$SRC_CONFIG" && "$SRC_CONFIG" -nt "$CONFIG" ]]; then
  echo "WARNING: $CONFIG is older than $SRC_CONFIG." >&2
  echo "         Re-run setup_offline_dinov3.sh so batch/epoch/centering match the source." >&2
fi

# Guard against the failure mode that invalidated the 55k-step run: Sinkhorn
# with a 16-row CLS batch produces a uniform teacher target (loss == ln K).
if [[ "$ALLOW_SINKHORN" != "1" ]] && ! grep -Eq '^[[:space:]]*centering:[[:space:]]*"?softmax"?' "$CONFIG"; then
  echo "ERROR: $CONFIG does not set train.centering: softmax." >&2
  echo "       Set ALLOW_SINKHORN=1 only if you really know the batch is large." >&2
  exit 1
fi

if [[ "$SMOKE" == "1" ]]; then
  OUTPUT="${OUTPUT}-smoke"
  RESUME=0
fi
mkdir -p "$OUTPUT"

has_checkpoint() {
  find "$OUTPUT/ckpt" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null | grep -q .
}

if [[ "$RESUME" == "1" ]]; then
  if ! has_checkpoint; then
    echo "ERROR: RESUME=1 but no checkpoint found under $OUTPUT/ckpt" >&2
    echo "       Run Stage 1 first, or point OUTPUT at the Stage 1 directory." >&2
    exit 1
  fi
elif [[ "$SMOKE" != "1" ]] && has_checkpoint && [[ "${OVERWRITE:-0}" != "1" ]]; then
  echo "ERROR: $OUTPUT already contains checkpoints." >&2
  echo "       Use a new OUTPUT directory, or set OVERWRITE=1 to start over." >&2
  exit 1
fi

# Run from lab_train so a relative ``./dataset`` in the config resolves to
# lab_train/dataset (the official repo used to be the cwd, which broke it).
cd "$LAB_DIR"

if [[ "$RESUME" == "1" ]]; then
  ARGS=(torchrun --nproc_per_node="$NPROC" third_party/dinov3/dinov3/train/train.py
        --config-file "$CONFIG" --output-dir "$OUTPUT")
else
  ARGS=(torchrun --nproc_per_node="$NPROC" third_party/dinov3/dinov3/train/train.py
        --config-file "$CONFIG" --output-dir "$OUTPUT" --no-resume)
fi

if [[ "$SMOKE" == "1" ]]; then
  # Keep every schedule length equal to the 100 smoke iterations:
  # warmup must not exceed the single smoke epoch.
  ARGS+=(
    train.OFFICIAL_EPOCH_LENGTH=100
    optim.epochs=1
    optim.warmup_epochs=0
    teacher.warmup_teacher_temp_epochs=1
    optim.freeze_last_layer_epochs=0
  )
fi

echo "config : $CONFIG"
echo "output : $OUTPUT"
echo "nproc  : $NPROC"
echo "smoke  : $SMOKE"
echo "resume : $RESUME"
grep -E '^[[:space:]]*(centering|batch_size_per_gpu|OFFICIAL_EPOCH_LENGTH|dataset_path|epochs|lr|head_n_prototypes|use_loss):' "$CONFIG" || true
"${ARGS[@]}" "$@" 2>&1 | tee -a "$OUTPUT/stdout.log"
