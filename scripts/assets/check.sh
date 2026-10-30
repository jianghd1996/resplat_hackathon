#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEIGHT_NAME=resplat-base-dl3dv-540x960-view16-a72dc6d0.pth
WEIGHT_SHA256=a72dc6d0d56f7ccc50d145c17c83d866d4aaa2c13adb1a22f7a84481001e1a4c
WEIGHT="$ROOT/resplat/pretrained/$WEIGHT_NAME"

if [ ! -f "$WEIGHT" ]; then
  echo "Missing model weight: $WEIGHT" >&2
  exit 1
fi
echo "$WEIGHT_SHA256  $WEIGHT" | sha256sum --check --status || {
  echo "Model weight SHA256 mismatch: $WEIGHT" >&2
  exit 1
}

python3 - "$ROOT" "$@" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "baseline/scenes.json").read_text())
groups = sys.argv[2:] or list(config)
missing = []
for group in groups:
    if group not in config:
        missing.append(f"Unknown group: {group}")
        continue
    scene = config[group]
    path = root / scene["source"] / scene["scene"]
    if path.is_dir():
        print(f"{group}: {path}")
    else:
        missing.append(f"Missing {group} scene: {path}")
if missing:
    print("\n".join(missing), file=sys.stderr)
    sys.exit(1)
print("Assets ready")
PY
