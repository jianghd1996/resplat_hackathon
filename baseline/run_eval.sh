#!/usr/bin/env bash
# One-click evaluation:
#   1. render held-out test views with Module B
#   2. compute PSNR / SSIM / LPIPS
#   3. write the timing and metric summary
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ITERATIONS=${ITERATIONS:-10000}

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
  WORK="$HERE/work/$GRP/$SCENE"
  RES="$HERE/results/$GRP/$SCENE"
  MODEL="$RES/module_b"
  if [ ! -f "$MODEL/point_cloud/iteration_$ITERATIONS/point_cloud.ply" ]; then
    echo "[$GRP/$SCENE] trained model missing ($MODEL) - skipping"
    continue
  fi

  echo ""
  echo "=== [$GRP] $SCENE : render + eval ==="

  activate_env "$FASTGS_ENV"
  python "$HERE/render_test.py" \
      -s "$WORK" \
      -m "$MODEL" \
      --eval -r 1 \
      --backend fastgs \
      --iteration "$ITERATIONS" \
      --quiet

  activate_env "$RESPLAT_ENV"
  python "$HERE/evaluate.py" --model_path "$MODEL"
done

activate_env "$RESPLAT_ENV"
python "$HERE/summarize.py" --work "$HERE/work" --results "$HERE/results" --out "$HERE/results"
