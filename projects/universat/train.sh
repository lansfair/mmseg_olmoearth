#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMSEG_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
CONFIG="${1:-projects/universat/pastis/configs/universat-base_pastis_lp.py}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ "$CONFIG" != /* ]]; then
    CONFIG="$MMSEG_ROOT/$CONFIG"
fi

CONFIG_NAME="$(basename "${CONFIG%.py}")"
WORK_DIR="${WORK_DIR:-$MMSEG_ROOT/work_dirs/$CONFIG_NAME}"
PYTHON_BIN="${PYTHON:-python3}"

cd "$MMSEG_ROOT"
export PYTHONPATH="$MMSEG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" tools/train.py "$CONFIG" --work-dir "$WORK_DIR" "$@"
