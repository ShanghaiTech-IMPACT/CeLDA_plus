#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_PATH="${DATA_PATH:-$REPO_DIR/data/CephaAdoAdu201}"
RUN_DIR="${RUN_DIR:?Please set RUN_DIR to the smoke-train output folder.}"
CHECKPOINT="${CHECKPOINT:-best_ema}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

cd "$REPO_DIR"
"$PYTHON_BIN" -u code/test.py \
  --run_dir "$RUN_DIR" \
  --data_path "$DATA_PATH" \
  --number_of_keypoints 201 \
  --checkpoint "$CHECKPOINT"
