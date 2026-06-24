#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_PATH="${DATA_PATH:-$REPO_DIR/data/CephaAdoAdu201}"
SAVE_PATH="${SAVE_PATH:-$REPO_DIR/workdir}"
EXP_NAME="${EXP_NAME:-CeLDA_201}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

cd "$REPO_DIR"
"$PYTHON_BIN" -u code/train.py \
  --data_path "$DATA_PATH" \
  --save_path "$SAVE_PATH" \
  --exp "$EXP_NAME" \
  --number_of_keypoints 201 \
  --max_epochs 100 \
  --batch_size 4\
  --base_lr 0.001

