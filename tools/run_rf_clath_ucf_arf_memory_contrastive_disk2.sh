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
ARF_POSITIVE_TOPK="${ARF_POSITIVE_TOPK:-10}"
ARF_POSITIVE_THRESHOLD="${ARF_POSITIVE_THRESHOLD:-0.0}"
ARF_INCLUDE_MISSED_AS_POSITIVE="${ARF_INCLUDE_MISSED_AS_POSITIVE:-false}"
ARF_HARD_POSITIVE_WEIGHT="${ARF_HARD_POSITIVE_WEIGHT:-1.0}"
ARF_HARD_NEGATIVE_WEIGHT="${ARF_HARD_NEGATIVE_WEIGHT:-1.5}"
ARF_ACTUAL_TRACE_START_EPOCH="${ARF_ACTUAL_TRACE_START_EPOCH:-0}"
ARF_HARD_MINING_START_EPOCH="${ARF_HARD_MINING_START_EPOCH:-0}"
TAG="${TAG:-arf_mem_contrastive}"

CONFIG=configs/rf_clath_ucf.yaml
DATASET=s5vh_ucf
OUTPUT_DIR="${OUTPUT_ROOT}/rf_clath_${TAG}_ucf_disk2"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

run_ucf() {
  cd "$PROJECT_ROOT"

  for bits in $BITS; do
    echo "$(timestamp) | rf_clath_${TAG}_ucf ${bits}-bit start on cuda${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" train.py \
      --config "$CONFIG" \
      --dataset "$DATASET" \
      --device cuda \
      --output-dir "$OUTPUT_DIR" \
      --epochs 150 \
      --hash-bits "$bits" \
      --override "project.name=RF-CLaTH-ARF-Memory-Contrastive-UCF-RePartition" \
      --override "training.objective=arf_memory_contrastive" \
      --override "train.objective=arf_memory_contrastive" \
      --override "loss.type=arf_memory_contrastive" \
      --override "loss.view.lambda=${VIEW_LAMBDA}" \
      --override "loss.semantic.lambda_batch_neighbor=${BATCH_LAMBDA}" \
      --override "loss.semantic.lambda_memory_neighbor=0.0" \
      --override "loss.arf.lambda=${ARF_LAMBDA}" \
      --override "loss_weights.lambda_arf=${ARF_LAMBDA}" \
      --override "arf_contrastive.positive_topk=${ARF_POSITIVE_TOPK}" \
      --override "arf_contrastive.positive_threshold=${ARF_POSITIVE_THRESHOLD}" \
      --override "arf_contrastive.include_missed_as_positive=${ARF_INCLUDE_MISSED_AS_POSITIVE}" \
      --override "arf_contrastive.hard_positive_weight=${ARF_HARD_POSITIVE_WEIGHT}" \
      --override "arf_contrastive.actual_trace_start_epoch=${ARF_ACTUAL_TRACE_START_EPOCH}" \
      --override "arf_contrastive.hard_mining_start_epoch=${ARF_HARD_MINING_START_EPOCH}" \
      --override "arf_contrastive.hard_negative_weight=${ARF_HARD_NEGATIVE_WEIGHT}"
  done

  echo "$(timestamp) | rf_clath_${TAG}_ucf done"
}

log_file="${LOG_ROOT}/rf_clath_${TAG}_ucf_disk2_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | RF-CLaTH ARF memory contrastive UCF run start, bits=${BITS}, gpu=${GPU}, view=${VIEW_LAMBDA}, batch=${BATCH_LAMBDA}, arf=${ARF_LAMBDA}, pos_topk=${ARF_POSITIVE_TOPK}, missed_pos=${ARF_INCLUDE_MISSED_AS_POSITIVE}, hard_pos_w=${ARF_HARD_POSITIVE_WEIGHT}, hard_neg_w=${ARF_HARD_NEGATIVE_WEIGHT}, actual_start=${ARF_ACTUAL_TRACE_START_EPOCH}, hard_start=${ARF_HARD_MINING_START_EPOCH}, log=${log_file}"
run_ucf >> "$log_file" 2>&1
echo "$(timestamp) | RF-CLaTH ARF memory contrastive UCF run done"
