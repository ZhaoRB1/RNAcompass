#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_environment.sh"
MODEL="${1:-ernie_rna_giga}"
GPU="${2:-0}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then shift; fi
exec "$PYTHON_BIN" -u "$TRAINING_DIR/train.py" \
    --config "$TRAINING_DIR/configs/r2/$MODEL.json" --gpu_id "$GPU" "$@"
