#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

GPU="${1:-3}"
BITS="${BITS:-16}"
STAGE1_WARMUP_EPOCHS="${STAGE1_WARMUP_EPOCHS:-30}"
SWITCH_EPOCH=$((STAGE1_WARMUP_EPOCHS + 1))
TAG="${TAG:-stage1warm${STAGE1_WARMUP_EPOCHS}_agentic_unified_v1}"

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
      --override "project.name=RF-CLaTH-Stage1Warmup-AgenticUnified-HMDB-RePartition" \
      --override "training.objective=stage1_warmup_agentic_unified" \
      --override "train.objective=stage1_warmup_agentic_unified" \
      --override "loss.type=stage1_warmup_agentic_unified" \
      --override "loss.hash.lambda_quant=0.02" \
      --override "loss.hash.lambda_bit_balance=0.03" \
      --override "loss.memory_neighbor.positives_per_anchor=15" \
      --override "loss.memory_neighbor.start_epoch=2" \
      --override "agentic_contrastive.source_weights.view=1.0" \
      --override "agentic_contrastive.source_weights.batch_neighbor=0.75" \
      --override "agentic_contrastive.source_weights.memory_neighbor=0.25" \
      --override "agentic_contrastive.source_weights.arf_planned=0.25" \
      --override "agentic_contrastive.source_weights.arf_missed_bonus=0.25" \
      --override "agentic_contrastive.hard_negative_weight=1.25" \
      --override "agentic_contrastive.stage1_warmup_epochs=${STAGE1_WARMUP_EPOCHS}" \
      --override "agentic_contrastive.actual_trace_start_epoch=${SWITCH_EPOCH}" \
      --override "agentic_contrastive.hard_mining_start_epoch=${SWITCH_EPOCH}"
  done

  echo "$(timestamp) | rf_clath_${TAG}_hmdb done"
}

log_file="${LOG_ROOT}/rf_clath_${TAG}_hmdb_disk2_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | RF-CLaTH Stage1 warmup -> Agentic Unified HMDB run start, bits=${BITS}, gpu=${GPU}, warmup_epochs=${STAGE1_WARMUP_EPOCHS}, switch_epoch=${SWITCH_EPOCH}, log=${log_file}"
run_hmdb >> "$log_file" 2>&1
echo "$(timestamp) | RF-CLaTH Stage1 warmup -> Agentic Unified HMDB run done"
