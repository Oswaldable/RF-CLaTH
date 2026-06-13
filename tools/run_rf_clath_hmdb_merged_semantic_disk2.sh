#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

GPU="${1:-3}"
BITS="${BITS:-16}"
EPOCHS="${EPOCHS:-150}"

CONFIG=configs/rf_clath_hmdb.yaml
DATASET=hmdb
OUTPUT_DIR="${OUTPUT_ROOT}/rf_clath_hmdb_merged_semantic_remaining_fast_disk2"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

run_hmdb() {
  cd "$PROJECT_ROOT"

  for bits in $BITS; do
    echo "$(timestamp) | rf_clath_hmdb_merged_semantic ${bits}-bit start on cuda${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" train.py \
      --config "$CONFIG" \
      --dataset "$DATASET" \
      --device cuda \
      --output-dir "$OUTPUT_DIR" \
      --epochs "$EPOCHS" \
      --hash-bits "$bits" \
      --override "project.name=RF-CLaTH-HMDB-MergedSemantic-RemainingFast" \
      --override "training.objective=merged_semantic" \
      --override "train.objective=merged_semantic" \
      --override "loss.type=merged_semantic" \
      --override "model.fast_encoder.input_frames=remaining" \
      --override "loss.semantic.lambda_merged=0.8" \
      --override "loss.semantic.view_positive_weight=0.6" \
      --override "loss.semantic.neighbor_positive_weight=1.0" \
      --override "loss.semantic.max_positive_weight=2.0" \
      --override "loss.semantic.lambda_memory_neighbor=0.04"
  done

  echo "$(timestamp) | rf_clath_hmdb_merged_semantic done"
}

log_file="${LOG_ROOT}/rf_clath_hmdb_merged_semantic_cuda${GPU}_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | RF-CLaTH HMDB merged-semantic run start, bits=${BITS}, gpu=${GPU}, log=${log_file}"
run_hmdb >> "$log_file" 2>&1
echo "$(timestamp) | RF-CLaTH HMDB merged-semantic run done"
