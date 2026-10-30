#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEIGHT_NAME=resplat-base-dl3dv-540x960-view16-a72dc6d0.pth
WEIGHT_SHA256=a72dc6d0d56f7ccc50d145c17c83d866d4aaa2c13adb1a22f7a84481001e1a4c
WEIGHT_URL="https://huggingface.co/haofeixu/resplat/resolve/main/$WEIGHT_NAME"
DATA_URL="https://drive.google.com/drive/folders/1CZWTj9jHh9wVdUru0za69-Mtaz5Fzn16?usp=drive_link"

download_weight() {
  command -v curl >/dev/null || { echo 'curl is required' >&2; exit 1; }
  mkdir -p "$ROOT/pretrained"
  local target="$ROOT/pretrained/$WEIGHT_NAME"
  local partial="$target.part"

  if [ -f "$target" ]; then
    echo "$WEIGHT_SHA256  $target" | sha256sum --check --status || {
      echo "Existing weight has the wrong SHA256: $target" >&2
      exit 1
    }
  else
    curl --fail --location --retry 3 --retry-delay 2 --continue-at - \
      --output "$partial" "$WEIGHT_URL"
    if ! echo "$WEIGHT_SHA256  $partial" | sha256sum --check --status; then
      rm -f "$partial"
      echo 'Downloaded weight failed SHA256 verification' >&2
      exit 1
    fi
    mv "$partial" "$target"
  fi

  local model_dir="$ROOT/resplat/pretrained"
  if [ ! -e "$model_dir" ] && [ ! -L "$model_dir" ]; then
    ln -s ../pretrained "$model_dir"
  fi
  if [ -d "$model_dir" ] && [ ! -e "$model_dir/$WEIGHT_NAME" ]; then
    ln -s "../../pretrained/$WEIGHT_NAME" "$model_dir/$WEIGHT_NAME"
  fi
  echo "Weight ready: $target"
}

download_datasets() {
  command -v gdown >/dev/null || { echo 'gdown is required: python -m pip install gdown' >&2; exit 1; }
  mkdir -p "$ROOT/datasets"
  gdown --folder --continue --output "$ROOT/datasets" "${DATASET_URL:-$DATA_URL}"

  # Some copies of the organizer folder contain an examples/ parent directory.
  for group in DL3DV hw mipnerf360; do
    if [ ! -e "$ROOT/datasets/$group" ] && [ -d "$ROOT/datasets/examples/$group" ]; then
      ln -s "examples/$group" "$ROOT/datasets/$group"
    fi
  done
  echo "Dataset download finished: $ROOT/datasets"
  echo 'Run bash scripts/assets/check.sh after downloading the weight.'
}

case "${1:-all}" in
  weight) download_weight ;;
  datasets) download_datasets ;;
  all) download_weight; download_datasets ;;
  *) echo 'Usage: bash scripts/assets/download.sh [weight|datasets|all]' >&2; exit 2 ;;
esac
