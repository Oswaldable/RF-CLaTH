#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

GPU="${1:-2}"
BITS="${BITS:-16}"
STAGE1_WARMUP_EPOCHS="${STAGE1_WARMUP_EPOCHS:-60}"
SCHEDULE_MODE="${SCHEDULE_MODE:-hard}"
RAMP_EPOCHS="${RAMP_EPOCHS:-0}"
POST_STAGE1_WEIGHT="${POST_STAGE1_WEIGHT:-0.0}"
RUN_UNTIL_EPOCH="${RUN_UNTIL_EPOCH:-}"
TAG="${TAG:-stage1warm${STAGE1_WARMUP_EPOCHS}_${SCHEDULE_MODE}_agentic_unified_v1}"
SWITCH_EPOCH=$((STAGE1_WARMUP_EPOCHS + 1))

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
      --override "project.name=RF-CLaTH-Stage1Scheduled-AgenticUnified-HMDB-RePartition"
      --override "training.objective=stage1_scheduled_agentic_unified"
      --override "train.objective=stage1_scheduled_agentic_unified"
      --override "loss.type=stage1_scheduled_agentic_unified"
      --override "loss.hash.lambda_quant=0.02"
      --override "loss.hash.lambda_bit_balance=0.03"
      --override "loss.memory_neighbor.positives_per_anchor=15"
      --override "loss.memory_neighbor.start_epoch=2"
      --override "agentic_contrastive.source_weights.view=1.0"
      --override "agentic_contrastive.source_weights.batch_neighbor=0.75"
      --override "agentic_contrastive.source_weights.memory_neighbor=0.25"
      --override "agentic_contrastive.source_weights.arf_planned=0.25"
      --override "agentic_contrastive.source_weights.arf_missed_bonus=0.25"
      --override "agentic_contrastive.hard_negative_weight=1.25"
      --override "agentic_contrastive.stage1_warmup_epochs=${STAGE1_WARMUP_EPOCHS}"
      --override "agentic_contrastive.schedule_mode=${SCHEDULE_MODE}"
      --override "agentic_contrastive.ramp_epochs=${RAMP_EPOCHS}"
      --override "agentic_contrastive.post_stage1_weight=${POST_STAGE1_WEIGHT}"
      --override "agentic_contrastive.actual_trace_start_epoch=${SWITCH_EPOCH}"
      --override "agentic_contrastive.hard_mining_start_epoch=${SWITCH_EPOCH}"
    )
    if [[ -n "$RUN_UNTIL_EPOCH" ]]; then
      cmd+=(--override "train.run_until_epoch=${RUN_UNTIL_EPOCH}")
    fi
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "${cmd[@]}"
  done

  echo "$(timestamp) | rf_clath_${TAG}_hmdb done"
}

log_file="${LOG_ROOT}/rf_clath_${TAG}_hmdb_disk2_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | RF-CLaTH Stage1 scheduled -> Agentic Unified HMDB run start, bits=${BITS}, gpu=${GPU}, warmup_epochs=${STAGE1_WARMUP_EPOCHS}, mode=${SCHEDULE_MODE}, ramp_epochs=${RAMP_EPOCHS}, post_stage1_weight=${POST_STAGE1_WEIGHT}, switch_epoch=${SWITCH_EPOCH}, run_until=${RUN_UNTIL_EPOCH:-150}, log=${log_file}"
run_hmdb >> "$log_file" 2>&1
echo "$(timestamp) | RF-CLaTH Stage1 scheduled -> Agentic Unified HMDB run done"
