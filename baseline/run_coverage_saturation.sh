#!/usr/bin/env bash
# Analyze geometric coverage saturation from an existing ReSplat-only ablation.
# This is CPU-only: it does not rerun ReSplat, rendering, or evaluation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ENV_PATH="${ENV_PATH:-/mnt/DataPart/jianghongda/env/resplat}"
ABLATION_ROOT="${ABLATION_ROOT:-$REPO/results/context_selection_hw}"
OUT="${OUT:-$ABLATION_ROOT/coverage_saturation}"
MAX_POINTS="${MAX_POINTS:-140000}"

if [ ! -x "$ENV_PATH/bin/python" ]; then
  echo "[ERROR] Python environment missing: $ENV_PATH" >&2
  exit 1
fi
if [ ! -d "$ABLATION_ROOT/runs" ]; then
  echo "[ERROR] ablation runs missing: $ABLATION_ROOT/runs" >&2
  exit 1
fi

echo "[coverage] ablation root: $ABLATION_ROOT"
echo "[coverage] output:        $OUT"
echo "[coverage] max points:    $MAX_POINTS"
echo "[coverage] GPU:           not used"

"$ENV_PATH/bin/python" "$HERE/analyze_coverage_saturation.py" \
  --ablation-root "$ABLATION_ROOT" \
  --out "$OUT" \
  --max-points "$MAX_POINTS" \
  "$@"

echo
echo "[done] report:  $OUT/report.md"
echo "[done] summary: $OUT/coverage_saturation_summary.png"
echo "[done] curves:  $OUT/coverage_curves.csv"
