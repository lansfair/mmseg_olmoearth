#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-projects/universat/pastis/configs/universat-base_pastis_lp.py}"
if [[ $# -gt 0 ]]; then
    shift
fi
exec "$PROJECT_DIR/../train.sh" "$CONFIG" "$@"
