#!/usr/bin/env bash
# One-time setup: install the CUDA extensions used by classic 3DGS and FastGS.
# Use regular wheels rather than editable wheels: simple-knn's legacy setup.py
# does not expose its extension namespace correctly through PEP 660 editable installs.
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FASTGS_UPSTREAM_COMMIT=${FASTGS_UPSTREAM_COMMIT:-44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f}
FASTGS_SOURCE=${FASTGS_SOURCE:-$HERE/.deps/FastGS}

# shellcheck disable=SC1091
source "$HERE/conda_env.sh"
activate_env "$FASTGS_ENV"
require_python_module torch

if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  else
    CUDA_HOME="$CONDA_PREFIX"
  fi
fi
export CUDA_HOME
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export CC=${CC:-/usr/bin/gcc}
export CXX=${CXX:-/usr/bin/g++}

echo "[setup] environment: $CONDA_DEFAULT_ENV"
echo "[setup] Python: $(command -v python)"
echo "[setup] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "[setup] CUDA_HOME: $CUDA_HOME"
echo "[setup] TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

install_extension() {
  local source_dir="$1"
  local label="$2"
  echo "=== installing $label ==="
  python -m pip install --no-build-isolation --no-deps --force-reinstall "$source_dir"
}

# Shared helpers and the classic renderer are vendored in this repository.
install_extension "$HERE/3dgs/submodules/simple-knn" "simple-knn"
install_extension "$HERE/3dgs/submodules/fused-ssim" "fused-ssim"
install_extension "$HERE/3dgs/submodules/diff-gaussian-rasterization" "classic diff-gaussian-rasterization"

# The imported baseline omitted FastGS's dedicated rasterizer sources. Fetch a
# pinned official FastGS revision so setup is reproducible.
if [ ! -d "$FASTGS_SOURCE/.git" ]; then
  mkdir -p "$(dirname "$FASTGS_SOURCE")"
  git clone https://github.com/fastgs/FastGS.git "$FASTGS_SOURCE"
fi
git -C "$FASTGS_SOURCE" fetch --depth 1 origin "$FASTGS_UPSTREAM_COMMIT"
git -C "$FASTGS_SOURCE" checkout --detach "$FASTGS_UPSTREAM_COMMIT"
install_extension "$FASTGS_SOURCE/submodules/diff-gaussian-rasterization_fastgs" \
                  "FastGS diff-gaussian-rasterization"

echo "=== validating CUDA extension imports ==="
python - <<'PY'
import torch
import simple_knn._C
import fused_ssim
import diff_gaussian_rasterization
import diff_gaussian_rasterization_fastgs

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("all CUDA extension imports: OK")
PY

echo "setup done."
