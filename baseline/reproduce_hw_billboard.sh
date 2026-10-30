#!/usr/bin/env bash
# Reproduce the current HW billboard baseline:
# uniform16 -> ReSplat (16 views) -> FastGS (10k) -> held-out evaluation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

GPU="${GPU:-6}"
ENV_PATH="${ENV_PATH:-/mnt/DataPart/jianghongda/env/resplat}"
NUM_CONTEXT="${NUM_CONTEXT:-16}"
ITERATIONS="${ITERATIONS:-10000}"
SCENE_DIR="$REPO/datasets/hw/billboard"
RESULT_DIR="$REPO/results/baseline_A_baseline_B/hw/billboard"
WEIGHT_NAME="resplat-base-dl3dv-540x960-view16-a72dc6d0.pth"
CHECKPOINT="${CHECKPOINT:-$REPO/resplat/pretrained/$WEIGHT_NAME}"

if [ ! -f "$CHECKPOINT" ] && [ -f "$REPO/pretrained/$WEIGHT_NAME" ]; then
  CHECKPOINT="$REPO/pretrained/$WEIGHT_NAME"
fi

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

[ -d "$SCENE_DIR" ] || fail "scene missing: $SCENE_DIR"
[ -f "$CHECKPOINT" ] || fail "checkpoint missing: $CHECKPOINT"
[ -x "$ENV_PATH/bin/python" ] || fail "Python environment missing: $ENV_PATH"

SPARSE="$SCENE_DIR/sparse/0"
if ! { [ -f "$SPARSE/cameras.bin" ] && [ -f "$SPARSE/images.bin" ]; } &&
   ! { [ -f "$SPARSE/cameras.txt" ] && [ -f "$SPARSE/images.txt" ]; }; then
  fail "valid COLMAP cameras/images pair missing under $SPARSE"
fi

if [ -f "$SPARSE/conversion_summary.json" ]; then
  fail "stale direct frames_info conversion detected: $SPARSE/conversion_summary.json"
fi

if [ -f "$SPARSE/pycolmap_undistorted.json" ] &&
   [ ! -d "$SCENE_DIR/images_undistorted" ]; then
  fail "undistorted COLMAP model requires $SCENE_DIR/images_undistorted"
fi

if [ -z "${CONDA_SH:-}" ] &&
   [ -f /home/jianghongda/miniconda3/etc/profile.d/conda.sh ]; then
  export CONDA_SH=/home/jianghongda/miniconda3/etc/profile.d/conda.sh
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs/reproduce_hw_billboard/$STAMP"
mkdir -p "$LOG_DIR"

if [ -e "$RESULT_DIR" ]; then
  BACKUP_DIR="${RESULT_DIR}.bak_$STAMP"
  mv "$RESULT_DIR" "$BACKUP_DIR"
  echo "[backup] $RESULT_DIR -> $BACKUP_DIR"
fi

export RESPLAT_ENV="$ENV_PATH"
export FASTGS_ENV="$ENV_PATH"
export CHECKPOINT
export NUM_CONTEXT
export ITERATIONS
export USE_FRAMES_INFO=0

echo "[config] repo:       $REPO"
echo "[config] scene:      $SCENE_DIR"
echo "[config] checkpoint: $CHECKPOINT"
echo "[config] environment:$ENV_PATH"
echo "[config] GPU:        $GPU"
echo "[config] context:    $NUM_CONTEXT"
echo "[config] iterations: $ITERATIONS"
echo "[config] logs:       $LOG_DIR"

CUDA_VISIBLE_DEVICES="$GPU" \
  bash "$HERE/run_train.sh" \
    --module-a uniform16 \
    --module-b fastgs \
    hw \
  2>&1 | tee "$LOG_DIR/train.log"

CUDA_VISIBLE_DEVICES="$GPU" \
  bash "$HERE/run_eval.sh" \
    --module-a uniform16 \
    --module-b fastgs \
    hw \
  2>&1 | tee "$LOG_DIR/eval.log"

METRICS="$RESULT_DIR/module_b/metrics.json"
SUMMARY="$REPO/results/baseline_A_baseline_B/summary.md"

[ -f "$METRICS" ] || fail "evaluation finished without metrics: $METRICS"

echo
echo "================ HW billboard metrics ================"
cat "$METRICS"

if [ -f "$SUMMARY" ]; then
  echo
  echo "================ Experiment summary =================="
  cat "$SUMMARY"
fi

echo
echo "[done] result:  $RESULT_DIR"
echo "[done] metrics: $METRICS"
echo "[done] logs:    $LOG_DIR"
