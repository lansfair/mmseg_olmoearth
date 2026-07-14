#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-projects/universat/pastis/configs/universat-base_pastis_lp.py}"
CHECKPOINT_PATH="${2:-${CHECKPOINT:-}}"
if [[ $# -ge 2 ]]; then
    shift 2
elif [[ $# -eq 1 ]]; then
    shift
fi
exec "$PROJECT_DIR/../test.sh" "$CONFIG" "$CHECKPOINT_PATH" "$@"
