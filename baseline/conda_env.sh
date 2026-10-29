#!/usr/bin/env bash
# Shared Conda bootstrap for the baseline launch scripts.
#
# Optional overrides:
#   CONDA_SH=/path/to/conda.sh
#   RESPLAT_ENV=resplat
#   FASTGS_ENV=fastgs
#
# By default Module B reuses the existing resplat environment. If a dedicated
# FastGS environment is available, run with FASTGS_ENV=fastgs.

RESPLAT_ENV="${RESPLAT_ENV:-resplat}"
FASTGS_ENV="${FASTGS_ENV:-$RESPLAT_ENV}"

if [ -n "${CONDA_SH:-}" ]; then
  if [ ! -f "$CONDA_SH" ]; then
    echo "CONDA_SH does not exist: $CONDA_SH" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "$CONDA_SH"
elif [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
  eval "$("$CONDA_EXE" shell.bash hook)"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Conda was not found in this shell." >&2
  echo "Run 'conda init bash' and reopen the shell, or set CONDA_SH=/path/to/conda.sh." >&2
  return 1
fi

activate_env() {
  local env_name="$1"
  if ! conda activate "$env_name"; then
    echo "Failed to activate Conda environment: $env_name" >&2
    echo "Available environments:" >&2
    conda env list >&2 || true
    return 1
  fi
  echo "[env] active: $env_name ($CONDA_PREFIX)"
}
