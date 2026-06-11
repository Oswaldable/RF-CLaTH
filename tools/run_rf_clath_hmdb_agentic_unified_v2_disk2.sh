#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

GPU="${1:-2}"
BITS="${BITS:-16}"
STAGE1_WARMUP_EPOCHS="${STAGE1_WARMUP_EPOCHS:-60}"
AGENTIC_RAMP_EPOCHS="${AGENTIC_RAMP_EPOCHS:-20}"
MAX_AGENTIC_MEMORY_MIX="${MAX_AGENTIC_MEMORY_MIX:-0.25}"
ACTUAL_TRACE_START_EPOCH="${ACTUAL_TRACE_START_EPOCH:-61}"
HARD_MINING_START_EPOCH="${HARD_MINING_START_EPOCH:-80}"
PLANNED_POSITIVE_TOPK="${PLANNED_POSITIVE_TOPK:-5}"
MISSED_POSITIVE_TOPK="${MISSED_POSITIVE_TOPK:-5}"
RAW_POSITIVE_WEIGHT="${RAW_POSITIVE_WEIGHT:-1.0}"
PLANNED_POSITIVE_WEIGHT="${PLANNED_POSITIVE_WEIGHT:-0.5}"
MISSED_POSITIVE_WEIGHT="${MISSED_POSITIVE_WEIGHT:-1.25}"
HARD_NEGATIVE_WEIGHT="${HARD_NEGATIVE_WEIGHT:-1.10}"
RUN_UNTIL_EPOCH="${RUN_UNTIL_EPOCH:-}"
TAG="${TAG:-agentic_unified_v2}"

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
    cmd=(
      "$PYTHON_BIN" train.py
      --config "$CONFIG"
      --dataset "$DATASET"
      --device cuda
      --output-dir "$OUTPUT_DIR"
      --epochs 150
      --hash-bits "$bits"
      --override "project.name=RF-CLaTH-AgenticUnifiedV2-HMDB-RePartition"
      --override "training.objective=agentic_unified_contrastive_v2"
      --override "train.objective=agentic_unified_contrastive_v2"
      --override "loss.type=agentic_unified_contrastive_v2"
      --override "loss.semantic.lambda_batch_neighbor=0.5"
      --override "loss.semantic.lambda_memory_neighbor=0.04"
      --override "loss.hash.lambda_quant=0.02"
      --override "loss.hash.lambda_bit_balance=0.03"
      --override "loss.memory_neighbor.start_epoch=2"
      --override "loss.memory_neighbor.positives_per_anchor=10"
      --override "agentic_contrastive_v2.stage1_warmup_epochs=${STAGE1_WARMUP_EPOCHS}"
      --override "agentic_contrastive_v2.agentic_ramp_epochs=${AGENTIC_RAMP_EPOCHS}"
      --override "agentic_contrastive_v2.max_agentic_memory_mix=${MAX_AGENTIC_MEMORY_MIX}"
      --override "agentic_contrastive_v2.actual_trace_start_epoch=${ACTUAL_TRACE_START_EPOCH}"
      --override "agentic_contrastive_v2.hard_mining_start_epoch=${HARD_MINING_START_EPOCH}"
      --override "agentic_contrastive_v2.planned_positive_topk=${PLANNED_POSITIVE_TOPK}"
      --override "agentic_contrastive_v2.missed_positive_topk=${MISSED_POSITIVE_TOPK}"
      --override "agentic_contrastive_v2.raw_positive_weight=${RAW_POSITIVE_WEIGHT}"
      --override "agentic_contrastive_v2.planned_positive_weight=${PLANNED_POSITIVE_WEIGHT}"
      --override "agentic_contrastive_v2.missed_positive_weight=${MISSED_POSITIVE_WEIGHT}"
      --override "agentic_contrastive_v2.hard_negative_weight=${HARD_NEGATIVE_WEIGHT}"
      --override "agentic_contrastive_v2.max_positive_weight=2.0"
      --override "agentic_contrastive_v2.normalize_sources=false"
    )
    if [[ -n "$RUN_UNTIL_EPOCH" ]]; then
      cmd+=(--override "train.run_until_epoch=${RUN_UNTIL_EPOCH}")
    fi
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "${cmd[@]}"
  done

  echo "$(timestamp) | rf_clath_${TAG}_hmdb done"
}

log_file="${LOG_ROOT}/rf_clath_${TAG}_hmdb_disk2_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | RF-CLaTH Agentic Unified v2 HMDB run start, bits=${BITS}, gpu=${GPU}, warmup=${STAGE1_WARMUP_EPOCHS}, ramp=${AGENTIC_RAMP_EPOCHS}, max_mix=${MAX_AGENTIC_MEMORY_MIX}, actual_start=${ACTUAL_TRACE_START_EPOCH}, hard_start=${HARD_MINING_START_EPOCH}, planned_topk=${PLANNED_POSITIVE_TOPK}, missed_topk=${MISSED_POSITIVE_TOPK}, hard_neg=${HARD_NEGATIVE_WEIGHT}, run_until=${RUN_UNTIL_EPOCH:-150}, log=${log_file}"
run_hmdb >> "$log_file" 2>&1
echo "$(timestamp) | RF-CLaTH Agentic Unified v2 HMDB run done"
