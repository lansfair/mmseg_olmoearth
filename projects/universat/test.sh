#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMSEG_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
CONFIG="${1:-projects/universat/pastis/configs/universat-base_pastis_lp.py}"
CHECKPOINT_PATH="${2:-${CHECKPOINT:-}}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Usage: $0 [config.py] checkpoint.pth [extra test.py arguments]" >&2
    echo "Alternatively set CHECKPOINT=/path/to/checkpoint.pth." >&2
    exit 2
fi
if [[ $# -ge 2 ]]; then
    shift 2
elif [[ $# -eq 1 ]]; then
    shift
fi

if [[ "$CONFIG" != /* ]]; then
    CONFIG="$MMSEG_ROOT/$CONFIG"
fi
if [[ "$CHECKPOINT_PATH" != /* ]]; then
    CHECKPOINT_PATH="$MMSEG_ROOT/$CHECKPOINT_PATH"
fi

CONFIG_NAME="$(basename "${CONFIG%.py}")"
WORK_DIR="${WORK_DIR:-$MMSEG_ROOT/work_dirs/$CONFIG_NAME}"
PYTHON_BIN="${PYTHON:-python3}"

cd "$MMSEG_ROOT"
export PYTHONPATH="$MMSEG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" tools/test.py \
    "$CONFIG" "$CHECKPOINT_PATH" --work-dir "$WORK_DIR" "$@"
