#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GT_PATH="${GT_PATH:-$REPO_DIR/data/CephaAdoAdu201/test/test_anno.json}"
RUN_DIR="${RUN_DIR:?Please set RUN_DIR to the same folder used in run_test.sh.}"
CHECKPOINT="${CHECKPOINT:-best_ema}"
EVAL_MODE="${EVAL_MODE:-mix}"
LINE_INDEX_PY="${LINE_INDEX_PY:-$REPO_DIR/code/utils/line_index.py}"

PRED_PATH="$RUN_DIR/eval_test_${CHECKPOINT}_${EVAL_MODE}/test_prediction_results.json"


cd "$REPO_DIR"
"$PYTHON_BIN" -u code/downstream_task.py \
  --number_of_keypoints 201 \
  --gt_path "$GT_PATH" \
  --pred_path "$PRED_PATH" \
  --line_index_py "$LINE_INDEX_PY"
