#!/usr/bin/env bash
# One-time setup: build vendored CUDA extensions.
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# shellcheck disable=SC1091
source "$HERE/conda_env.sh"
activate_env "$FASTGS_ENV"

export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export CC=${CC:-/usr/bin/gcc}
export CXX=${CXX:-/usr/bin/g++}

echo "[setup] environment: $FASTGS_ENV"
echo "[setup] CUDA_HOME: $CUDA_HOME"
echo "[setup] TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

for mod in diff-gaussian-rasterization simple-knn fused-ssim; do
  echo "=== building $mod ==="
  cd "$HERE/3dgs/submodules/$mod"
  python setup.py build_ext --inplace
done

echo "setup done."
