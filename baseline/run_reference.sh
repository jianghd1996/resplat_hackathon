#!/usr/bin/env bash
# OPTIONAL: classic 3DGS 30k reference from random frustum points.
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ITERATIONS=${REF_ITERATIONS:-30000}

# shellcheck disable=SC1091
source "$HERE/conda_env.sh"
activate_env "$FASTGS_ENV"

SCENE_GROUPS=("$@")
if [ ${#SCENE_GROUPS[@]} -eq 0 ]; then
  SCENE_GROUPS=($(python -c "import json;print(' '.join(json.load(open('$HERE/scenes.json')).keys()))"))
fi

for GRP in "${SCENE_GROUPS[@]}"; do
  SCENE=$(python -c "import json;print(json.load(open('$HERE/scenes.json'))['$GRP']['scene'])")
  WORK="$HERE/work/$GRP/$SCENE"
  RES="$HERE/results/$GRP/$SCENE/reference_3dgs"
  if [ ! -f "$WORK/split.json" ]; then
    echo "[$GRP/$SCENE] work dir missing ($WORK) - run run_train.sh first; skipping"
    continue
  fi

  echo ""
  echo "=== [$GRP] $SCENE : classic 3DGS reference (30k, random init) ==="
  export PYTHONPATH="$HERE/3dgs:$HERE/3dgs/submodules/diff-gaussian-rasterization:$HERE/3dgs/submodules/simple-knn:${PYTHONPATH:-}"
  python "$HERE/module_b/train_from_ply.py" \
      -s "$WORK" \
      -m "$RES" \
      --eval \
      -r 1 \
      --sh_degree 3 \
      --iterations "$ITERATIONS" \
      --position_lr_max_steps "$ITERATIONS" \
      --densify_until_iter 15000 \
      --init_random \
      --quiet
done

python "$HERE/summarize.py" --work "$HERE/work" --results "$HERE/results" --out "$HERE/results"
