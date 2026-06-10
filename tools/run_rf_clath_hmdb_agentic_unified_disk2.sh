#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

GPU="${1:-0}"
BITS="${BITS:-16}"
MEMORY_POSITIVE_TOPK="${MEMORY_POSITIVE_TOPK:-10}"
ARF_POSITIVE_TOPK="${ARF_POSITIVE_TOPK:-10}"
ACTUAL_TRACE_START_EPOCH="${ACTUAL_TRACE_START_EPOCH:-30}"
HARD_MINING_START_EPOCH="${HARD_MINING_START_EPOCH:-30}"
WEIGHT_VIEW="${WEIGHT_VIEW:-1.0}"
WEIGHT_BATCH="${WEIGHT_BATCH:-0.75}"
WEIGHT_MEMORY="${WEIGHT_MEMORY:-0.25}"
WEIGHT_ARF="${WEIGHT_ARF:-0.25}"
WEIGHT_MISSED_BONUS="${WEIGHT_MISSED_BONUS:-0.25}"
HARD_NEGATIVE_WEIGHT="${HARD_NEGATIVE_WEIGHT:-1.25}"
TAG="${TAG:-agentic_unified}"

CONFIG=configs/rf_clath_hmdb_agentic_unified.yaml
DATASET=hmdb
OUTPUT_DIR="${OUTPUT_ROOT}/rf_clath_${TAG}_hmdb_disk2"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

run_hmdb() {
  cd "$PROJECT_ROOT"

  for bits in $BITS; do
    echo "$(timestamp) | rf_clath_${TAG}_hmdb ${bits}-bit start on cuda${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" train.py \
      --config "$CONFIG" \
      --dataset "$DATASET" \
      --device cuda \
      --output-dir "$OUTPUT_DIR" \
      --epochs 150 \
      --hash-bits "$bits" \
      --override "project.name=RF-CLaTH-AgenticUnified-HMDB-RePartition" \
      --override "training.objective=agentic_unified_contrastive" \
      --override "train.objective=agentic_unified_contrastive" \
      --override "loss.type=agentic_unified_contrastive" \
      --override "agentic_contrastive.memory_positive_topk=${MEMORY_POSITIVE_TOPK}" \
      --override "agentic_contrastive.arf_positive_topk=${ARF_POSITIVE_TOPK}" \
      --override "agentic_contrastive.actual_trace_start_epoch=${ACTUAL_TRACE_START_EPOCH}" \
      --override "agentic_contrastive.hard_mining_start_epoch=${HARD_MINING_START_EPOCH}" \
      --override "agentic_contrastive.source_weights.view=${WEIGHT_VIEW}" \
      --override "agentic_contrastive.source_weights.batch_neighbor=${WEIGHT_BATCH}" \
      --override "agentic_contrastive.source_weights.memory_neighbor=${WEIGHT_MEMORY}" \
      --override "agentic_contrastive.source_weights.arf_planned=${WEIGHT_ARF}" \
      --override "agentic_contrastive.source_weights.arf_missed_bonus=${WEIGHT_MISSED_BONUS}" \
      --override "agentic_contrastive.hard_negative_weight=${HARD_NEGATIVE_WEIGHT}"
  done

  echo "$(timestamp) | rf_clath_${TAG}_hmdb done"
}

log_file="${LOG_ROOT}/rf_clath_${TAG}_hmdb_disk2_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | RF-CLaTH Agentic Unified HMDB run start, bits=${BITS}, gpu=${GPU}, memory_topk=${MEMORY_POSITIVE_TOPK}, arf_topk=${ARF_POSITIVE_TOPK}, actual_start=${ACTUAL_TRACE_START_EPOCH}, hard_start=${HARD_MINING_START_EPOCH}, w_view=${WEIGHT_VIEW}, w_batch=${WEIGHT_BATCH}, w_memory=${WEIGHT_MEMORY}, w_arf=${WEIGHT_ARF}, w_missed=${WEIGHT_MISSED_BONUS}, hard_neg=${HARD_NEGATIVE_WEIGHT}, log=${log_file}"
run_hmdb >> "$log_file" 2>&1
echo "$(timestamp) | RF-CLaTH Agentic Unified HMDB run done"
