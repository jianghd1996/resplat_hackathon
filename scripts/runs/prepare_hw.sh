#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/baseline/conda_env.sh"
activate_env "$RESPLAT_ENV"

python "$ROOT/baseline/rebuild_hw_sparse.py" \
  --root "$ROOT/datasets/hw" \
  --backup-dir "${HW_BACKUP_DIR:-$ROOT/datasets/hw_backup}" \
  "$@"
