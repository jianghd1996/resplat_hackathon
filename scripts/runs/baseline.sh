#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-all}"
if [ "$#" -gt 0 ]; then shift; fi
case "$MODE" in
  train|eval|all) ;;
  *) echo 'Usage: bash scripts/runs/baseline.sh [train|eval|all] [dl3dv|hw|mipnerf360 ...]' >&2; exit 2 ;;
esac

# The remote machine uses one working Conda environment named resplat.
export RESPLAT_ENV="${RESPLAT_ENV:-resplat}"
export FASTGS_ENV="${FASTGS_ENV:-resplat}"

"$ROOT/scripts/assets/check.sh" "$@"
case "$MODE" in
  train) bash "$ROOT/baseline/run_train.sh" "$@" ;;
  eval) bash "$ROOT/baseline/run_eval.sh" "$@" ;;
  all)
    bash "$ROOT/baseline/run_train.sh" "$@"
    bash "$ROOT/baseline/run_eval.sh" "$@"
    ;;
esac
