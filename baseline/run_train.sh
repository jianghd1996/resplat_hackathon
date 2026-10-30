#!/usr/bin/env bash
# One-click end-to-end baseline training:
#   Module A (pose acquisition + 16-view selection)
#     -> ReSplat (16-view feed-forward reconstruction)
#     -> Module B (FastGS continuation, 10k steps, all train views)
#
# Usage:
#   bash run_train.sh                 # all groups in scenes.json
#   bash run_train.sh dl3dv hw        # subset of groups
#
# Optional environment overrides:
#   RESPLAT_ENV=resplat FASTGS_ENV=fastgs bash run_train.sh
#   CONDA_SH=/path/to/conda.sh bash run_train.sh
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
COMP="$(cd "$HERE/.." && pwd)"
NUM_CONTEXT=${NUM_CONTEXT:-16}
ITERATIONS=${ITERATIONS:-10000}
USE_FRAMES_INFO=${USE_FRAMES_INFO:-0}
MODULE_A=uniform16
MODULE_B=fastgs
SCENE_GROUPS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --module-a) MODULE_A="${2:?--module-a needs a value}"; shift 2 ;;
    --module-b) MODULE_B="${2:?--module-b needs a value}"; shift 2 ;;
    --) shift; SCENE_GROUPS+=("$@"); break ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) SCENE_GROUPS+=("$1"); shift ;;
  esac
done
if [[ ! "$MODULE_A" =~ ^[a-z][a-z0-9_]*$ ]] || \
   [[ ! "$MODULE_B" =~ ^[a-z][a-z0-9_]*$ ]] || \
   [ ! -f "$HERE/utils_A/$MODULE_A.py" ] || \
   [ ! -f "$HERE/utils_B/$MODULE_B.py" ]; then
  echo "Unsupported module selection: A=$MODULE_A B=$MODULE_B" >&2
  exit 2
fi
A_LABEL=$MODULE_A
B_LABEL=$MODULE_B
[ "$MODULE_A" = "uniform16" ] && A_LABEL=baseline
[ "$MODULE_B" = "fastgs" ] && B_LABEL=baseline
[ "$MODULE_B" = "short_budget" ] && B_LABEL="short_budget${ITERATIONS}"
[ "$MODULE_B" = "short_schedule" ] && B_LABEL="short_schedule${ITERATIONS}"
[ "$MODULE_B" = "regularized" ] && B_LABEL="regularized${ITERATIONS}"
[ "$MODULE_B" = "no_reset" ] && B_LABEL="no_reset${ITERATIONS}"
[ "$MODULE_B" = "topk_growth" ] && B_LABEL="topk_growth${ITERATIONS}"
[ "$MODULE_B" = "coarse_to_fine" ] && B_LABEL="coarse_to_fine${ITERATIONS}"
[ "$MODULE_B" = "long_axis_split" ] && B_LABEL="long_axis_split${ITERATIONS}"
[ "$MODULE_B" = "perceptual_loss" ] && B_LABEL="perceptual_loss${ITERATIONS}"
EXPERIMENT="${A_LABEL}_A_${B_LABEL}_B"
RESULTS="$COMP/results/$EXPERIMENT"
WEIGHT_NAME=resplat-base-dl3dv-540x960-view16-a72dc6d0.pth
if [ -z "${CHECKPOINT:-}" ]; then
  CHECKPOINT="$COMP/resplat/pretrained/$WEIGHT_NAME"
  if [ ! -f "$CHECKPOINT" ] && [ -f "$COMP/pretrained/$WEIGHT_NAME" ]; then
    CHECKPOINT="$COMP/pretrained/$WEIGHT_NAME"
  fi
fi
if [ ! -f "$CHECKPOINT" ]; then
  echo "ReSplat checkpoint missing: $CHECKPOINT (set CHECKPOINT=/path/to/weight.pth)" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$HERE/conda_env.sh"
activate_env "$RESPLAT_ENV"

if [ ${#SCENE_GROUPS[@]} -eq 0 ]; then
  SCENE_GROUPS=($(python -c "import json;print(' '.join(json.load(open('$HERE/scenes.json')).keys()))"))
fi

for GRP in "${SCENE_GROUPS[@]}"; do
  activate_env "$RESPLAT_ENV"
  SCENE=$(python -c "import json;print(json.load(open('$HERE/scenes.json'))['$GRP']['scene'])")
  SRC_REL=$(python -c "import json;print(json.load(open('$HERE/scenes.json'))['$GRP']['source'])")
  SCENE_DIR="$COMP/$SRC_REL/$SCENE"
  RES="$RESULTS/$GRP/$SCENE"
  WORK="$RES/work"

  echo ""
  echo "==================================================================="
  echo "  [$GRP] $SCENE"
  echo "==================================================================="

  if [ ! -d "$SCENE_DIR" ]; then
    echo "scene dir not found: $SCENE_DIR (dataset not downloaded?) - skipping"
    continue
  fi

  # ---------------- Module A + ReSplat ----------------
  MODULE_A_EXTRA_ARGS=()
  if [ "$USE_FRAMES_INFO" = "1" ]; then
    MODULE_A_EXTRA_ARGS+=(--use_frames_info)
  fi
  python "$HERE/module_a/preprocess.py" \
      --scene_dir "$SCENE_DIR" \
      --group "$GRP" \
      --out "$WORK" \
      --module-a "$MODULE_A" \
      --num_context "$NUM_CONTEXT" \
      "${MODULE_A_EXTRA_ARGS[@]}"

  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python "$HERE/run_resplat.py" \
      --work "$WORK" \
      --out "$RES/resplat" \
      --checkpoint "$CHECKPOINT"

  # ---------------- Module B: FastGS ----------------
  activate_env "$FASTGS_ENV"
  python "$HERE/utils_B/$MODULE_B.py" train \
      --work "$WORK" --out "$RES/module_b" \
      --iterations "$ITERATIONS" \
      --init-ply "$RES/resplat/resplat_world.ply"
done

echo ""
echo "Training finished in $RESULTS. Run 'bash run_eval.sh' to compute metrics."
