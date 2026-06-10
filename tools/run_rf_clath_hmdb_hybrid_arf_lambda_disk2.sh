#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

GPU="${1:-0}"
BITS="${BITS:-16}"
ARF_LAMBDA="${ARF_LAMBDA:-0.04}"
BATCH_LAMBDA="${BATCH_LAMBDA:-0.50}"
VIEW_LAMBDA="${VIEW_LAMBDA:-0.30}"
TAG="${TAG:-lambda${ARF_LAMBDA//./p}}"

CONFIG=configs/rf_clath_hmdb_hybrid_arf.yaml
DATASET=hmdb
OUTPUT_DIR="${OUTPUT_ROOT}/rf_clath_hybrid_arf_${TAG}_hmdb_disk2"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

run_hmdb() {
  cd "$PROJECT_ROOT"

  for bits in $BITS; do
    echo "$(timestamp) | rf_clath_hybrid_arf_${TAG}_hmdb ${bits}-bit start on cuda${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" train.py \
      --config "$CONFIG" \
      --dataset "$DATASET" \
      --device cuda \
      --output-dir "$OUTPUT_DIR" \
      --epochs 150 \
      --hash-bits "$bits" \
      --override "project.name=RF-CLaTH-HybridARF-${TAG}-HMDB-RePartition" \
      --override "loss.view.lambda=${VIEW_LAMBDA}" \
      --override "loss.semantic.lambda_batch_neighbor=${BATCH_LAMBDA}" \
      --override "loss.semantic.lambda_memory_neighbor=0.0" \
      --override "loss.arf.lambda=${ARF_LAMBDA}" \
      --override "loss_weights.lambda_arf=${ARF_LAMBDA}"
  done

  echo "$(timestamp) | rf_clath_hybrid_arf_${TAG}_hmdb done"
}

log_file="${LOG_ROOT}/rf_clath_hybrid_arf_${TAG}_hmdb_disk2_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | RF-CLaTH Hybrid ARF ${TAG} HMDB run start, bits=${BITS}, gpu=${GPU}, view=${VIEW_LAMBDA}, batch=${BATCH_LAMBDA}, arf=${ARF_LAMBDA}, log=${log_file}"
run_hmdb >> "$log_file" 2>&1
echo "$(timestamp) | RF-CLaTH Hybrid ARF ${TAG} HMDB run done"
