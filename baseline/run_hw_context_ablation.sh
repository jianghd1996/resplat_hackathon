#!/usr/bin/env bash
# ReSplat-only context-selection ablation on HW scenes.
#
# The evaluation stage uses the FastGS renderer only as a PLY renderer.  It
# never runs FastGS training or changes the ReSplat Gaussians.
#
# Usage:
#   bash run_hw_context_ablation.sh
#   GPUS=4,5,6,7 bash run_hw_context_ablation.sh
#   bash run_hw_context_ablation.sh billboard trashbin
#   STAGE=infer bash run_hw_context_ablation.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

ENV_PATH="${ENV_PATH:-/mnt/DataPart/jianghongda/env/resplat}"
DATASET_ROOT="${DATASET_ROOT:-$REPO/datasets}"
OUTPUT="${OUTPUT:-$REPO/results/context_selection_hw}"
GPUS="${GPUS:-6,7}"
STAGE="${STAGE:-all}"
NUM_CONTEXT="${NUM_CONTEXT:-16}"
WEIGHT_NAME="resplat-base-dl3dv-540x960-view16-a72dc6d0.pth"
CHECKPOINT="${CHECKPOINT:-$REPO/resplat/pretrained/$WEIGHT_NAME}"

if [ ! -f "$CHECKPOINT" ] && [ -f "$REPO/pretrained/$WEIGHT_NAME" ]; then
  CHECKPOINT="$REPO/pretrained/$WEIGHT_NAME"
fi

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

[ -x "$ENV_PATH/bin/python" ] || fail "Python environment missing: $ENV_PATH"
[ -d "$DATASET_ROOT/hw" ] || fail "HW dataset missing: $DATASET_ROOT/hw"
[ -f "$CHECKPOINT" ] || fail "checkpoint missing: $CHECKPOINT"

case "$STAGE" in
  all|prepare|infer|evaluate|summarize) ;;
  *) fail "STAGE must be all, prepare, infer, evaluate, or summarize" ;;
esac

STRATEGIES=(
  uniform16
  random16
  center_fps_zero
  pose_fps
  orbit_fps
  oracle_test_nearest
  oracle_test_nearest_pairs
)

SCENE_ARGS=()
if [ "$#" -gt 0 ]; then
  SCENE_ARGS=(--scenes "$@")
fi

FORCE_ARGS=()
if [ "${FORCE:-0}" = "1" ]; then
  FORCE_ARGS=(--force)
fi

mkdir -p "$OUTPUT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG="$OUTPUT/logs/driver_${STAGE}_$STAMP.log"

echo "[config] dataset:    $DATASET_ROOT/hw"
echo "[config] output:     $OUTPUT"
echo "[config] stage:      $STAGE"
echo "[config] GPUs:       $GPUS"
echo "[config] context:    $NUM_CONTEXT"
echo "[config] checkpoint: $CHECKPOINT"
if [ "$#" -gt 0 ]; then
  echo "[config] scenes:     $*"
else
  echo "[config] scenes:     all HW scene directories"
fi
echo "[config] strategies: ${STRATEGIES[*]}"
echo "[config] Module B training: disabled"

"$ENV_PATH/bin/python" "$HERE/batch_context_ablation.py" \
  --dataset-root "$DATASET_ROOT" \
  --output "$OUTPUT" \
  --groups hw \
  --strategies "${STRATEGIES[@]}" \
  --num-context "$NUM_CONTEXT" \
  --stage "$STAGE" \
  --gpus "$GPUS" \
  --checkpoint "$CHECKPOINT" \
  --resplat-python "$ENV_PATH/bin/python" \
  --fastgs-python "$ENV_PATH/bin/python" \
  "${SCENE_ARGS[@]}" \
  "${FORCE_ARGS[@]}" \
  2>&1 | tee "$DRIVER_LOG"

echo
echo "[done] output:  $OUTPUT"
echo "[done] summary: $OUTPUT/summary.md"
echo "[done] CSV:     $OUTPUT/per_scene.csv"
echo "[done] log:     $DRIVER_LOG"
