#!/usr/bin/env bash
# One-click evaluation:
#   1. render held-out test views with Module B
#   2. compute PSNR / SSIM / LPIPS
#   3. write the timing and metric summary
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
COMP="$(cd "$HERE/.." && pwd)"
ITERATIONS=${ITERATIONS:-10000}
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
EXPERIMENT="${A_LABEL}_A_${B_LABEL}_B"
RESULTS="$COMP/results/$EXPERIMENT"

# shellcheck disable=SC1091
source "$HERE/conda_env.sh"
activate_env "$RESPLAT_ENV"

if [ ${#SCENE_GROUPS[@]} -eq 0 ]; then
  SCENE_GROUPS=($(python -c "import json;print(' '.join(json.load(open('$HERE/scenes.json')).keys()))"))
fi

for GRP in "${SCENE_GROUPS[@]}"; do
  activate_env "$RESPLAT_ENV"
  SCENE=$(python -c "import json;print(json.load(open('$HERE/scenes.json'))['$GRP']['scene'])")
  RES="$RESULTS/$GRP/$SCENE"
  WORK="$RES/work"
  MODEL="$RES/module_b"
  if [ ! -f "$MODEL/point_cloud/iteration_$ITERATIONS/point_cloud.ply" ]; then
    echo "[$GRP/$SCENE] trained model missing ($MODEL) - skipping"
    continue
  fi

  echo ""
  echo "=== [$GRP] $SCENE : render + eval ==="

  activate_env "$FASTGS_ENV"
  python "$HERE/utils_B/$MODULE_B.py" render \
      --work "$WORK" --out "$MODEL" --iterations "$ITERATIONS"

  activate_env "$RESPLAT_ENV"
  python "$HERE/evaluate.py" --model_path "$MODEL"
done

activate_env "$RESPLAT_ENV"
python "$HERE/summarize.py" --results "$RESULTS" --out "$RESULTS"
