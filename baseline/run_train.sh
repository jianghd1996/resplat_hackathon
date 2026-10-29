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

# shellcheck disable=SC1091
source "$HERE/conda_env.sh"
activate_env "$RESPLAT_ENV"

SCENE_GROUPS=("$@")
if [ ${#SCENE_GROUPS[@]} -eq 0 ]; then
  SCENE_GROUPS=($(python -c "import json;print(' '.join(json.load(open('$HERE/scenes.json')).keys()))"))
fi

for GRP in "${SCENE_GROUPS[@]}"; do
  activate_env "$RESPLAT_ENV"
  SCENE=$(python -c "import json;print(json.load(open('$HERE/scenes.json'))['$GRP']['scene'])")
  SRC_REL=$(python -c "import json;print(json.load(open('$HERE/scenes.json'))['$GRP']['source'])")
  SCENE_DIR="$COMP/$SRC_REL/$SCENE"
  WORK="$HERE/work/$GRP/$SCENE"
  RES="$HERE/results/$GRP/$SCENE"

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
      --num_context "$NUM_CONTEXT" \
      "${MODULE_A_EXTRA_ARGS[@]}"

  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python "$HERE/run_resplat.py" \
      --work "$WORK" \
      --out "$RES/resplat"

  # ---------------- Module B: FastGS ----------------
  activate_env "$FASTGS_ENV"
  export PYTHONPATH="$HERE/fastgs:${PYTHONPATH:-}"
  python "$HERE/module_b/train_from_ply_fastgs.py" \
      -s "$WORK" \
      -m "$RES/module_b" \
      --eval \
      -r 1 \
      --sh_degree 3 \
      --iterations "$ITERATIONS" \
      --init_ply "$RES/resplat/resplat_world.ply" \
      --quiet
done

echo ""
echo "Training finished. Run 'bash run_eval.sh' to compute metrics."
