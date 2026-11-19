#!/usr/bin/env bash
set -euo pipefail
TRAINING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(dirname "$TRAINING_DIR")"
export RNACOMPASS_DATA_DIR="${RNACOMPASS_DATA_DIR:-$PROJECT_DIR/data}"
export RNACOMPASS_RESULTS_DIR="${RNACOMPASS_RESULTS_DIR:-$PROJECT_DIR/results}"
export PYTHONPATH="$TRAINING_DIR:$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rnacompass-training-matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export LD_LIBRARY_PATH="${RNACOMPASS_LIBRARY_PATH:-${LD_LIBRARY_PATH:-}}"
PYTHON_BIN="${RNACOMPASS_PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python interpreter not found: %s. Activate the rnacompass environment or set RNACOMPASS_PYTHON.\n' "$PYTHON_BIN" >&2
    exit 2
fi
