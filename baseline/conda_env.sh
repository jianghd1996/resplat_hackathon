#!/usr/bin/env bash
# Shared Conda bootstrap for the baseline launch scripts.
#
# Optional overrides:
#   CONDA_SH=/path/to/conda.sh
#   RESPLAT_ENV=/path/to/env-or-env-name
#   FASTGS_ENV=/path/to/env-or-env-name
#
# If a resplat environment is already active, its absolute path is preferred.
# This avoids selecting the wrong environment when duplicate names exist.

_ACTIVE_RESPLAT_ENV=""
if [ -n "${CONDA_PREFIX:-}" ] && \
   { [ "${CONDA_DEFAULT_ENV:-}" = "resplat" ] || [ "$(basename "$CONDA_PREFIX")" = "resplat" ]; }; then
  _ACTIVE_RESPLAT_ENV="$CONDA_PREFIX"
fi

RESPLAT_ENV="${RESPLAT_ENV:-${_ACTIVE_RESPLAT_ENV:-resplat}}"
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
  echo "[env] active: $CONDA_DEFAULT_ENV ($CONDA_PREFIX)"
}

require_python_module() {
  local module_name="$1"
  if ! python -c "import $module_name" >/dev/null 2>&1; then
    echo "Python module '$module_name' is missing from $CONDA_PREFIX." >&2
    echo "Python executable: $(command -v python)" >&2
    return 1
  fi
}
