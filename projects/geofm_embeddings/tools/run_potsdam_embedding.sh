#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 MODEL SPLIT GPU [LIMIT]" >&2
  exit 2
fi

MODEL="$1"
SPLIT="$2"
GPU="$3"
LIMIT="${4:-}"

REPO=/mnt/ht2-nas2/EO_test/wyf/embedding_code/geofm_a100/src/mmseg_olmoearth
PYTHON=/mnt/ht2-nas2/EO_test/miniconda3/envs/geofm-olmoearth-cu121/bin/python
OUTPUT_ROOT=/mnt/ht2-nas2/EO_test/wyf/embedding_code/embedding/potsdam
CONFIG="$REPO/projects/geofm_embeddings/configs/potsdam/${MODEL}.py"
OUTPUT="$OUTPUT_ROOT/$MODEL"

if [ ! -f "$CONFIG" ]; then
  echo "Unknown model config: $CONFIG" >&2
  exit 2
fi
if [ "$SPLIT" != train ] && [ "$SPLIT" != val ]; then
  echo "Potsdam split must be train or val" >&2
  exit 2
fi

mkdir -p "$OUTPUT" "$OUTPUT_ROOT/logs"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  "$REPO/projects/geofm_embeddings/tools/extract_embeddings.py"
  "$CONFIG"
  "$OUTPUT"
  --split "$SPLIT"
  --mode dense
  --dense-format pt
  --save-labels
  --precision bf16
)
if [ -n "$LIMIT" ]; then
  ARGS+=(--limit "$LIMIT")
fi

exec "$PYTHON" "${ARGS[@]}"
