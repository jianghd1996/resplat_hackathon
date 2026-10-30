#!/usr/bin/env bash
# OPTIONAL: classic 3DGS 30k reference from random frustum points.
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
COMP="$(cd "$HERE/.." && pwd)"
ITERATIONS=${REF_ITERATIONS:-30000}
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
RESULTS="$COMP/results/${A_LABEL}_A_${B_LABEL}_B"

# shellcheck disable=SC1091
source "$HERE/conda_env.sh"
activate_env "$FASTGS_ENV"

if [ ${#SCENE_GROUPS[@]} -eq 0 ]; then
  SCENE_GROUPS=($(python -c "import json;print(' '.join(json.load(open('$HERE/scenes.json')).keys()))"))
fi

for GRP in "${SCENE_GROUPS[@]}"; do
  SCENE=$(python -c "import json;print(json.load(open('$HERE/scenes.json'))['$GRP']['scene'])")
  WORK="$RESULTS/$GRP/$SCENE/work"
  RES="$RESULTS/$GRP/$SCENE/reference_3dgs"
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

python "$HERE/summarize.py" --results "$RESULTS" --out "$RESULTS"
