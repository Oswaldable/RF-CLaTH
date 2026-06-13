#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

GPU="${1:-2}"
BITS="${BITS:-16}"
EPOCHS="${EPOCHS:-150}"
ACTUAL_TRACE_START_EPOCH="${ACTUAL_TRACE_START_EPOCH:-61}"
HARD_MINING_START_EPOCH="${HARD_MINING_START_EPOCH:-80}"

CONFIG=configs/rf_clath_hmdb.yaml
DATASET=hmdb
OUTPUT_DIR="${OUTPUT_ROOT}/rf_clath_hmdb_memory_self_calibrated_remaining_fast_disk2"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

run_hmdb() {
  cd "$PROJECT_ROOT"

  for bits in $BITS; do
    echo "$(timestamp) | rf_clath_hmdb_memory_self_calibrated ${bits}-bit start on cuda${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" train.py \
      --config "$CONFIG" \
      --dataset "$DATASET" \
      --device cuda \
      --output-dir "$OUTPUT_DIR" \
      --epochs "$EPOCHS" \
      --hash-bits "$bits" \
      --override "project.name=RF-CLaTH-HMDB-MemorySelfCalibrated-RemainingFast" \
      --override "training.objective=memory_self_calibrated" \
      --override "train.objective=memory_self_calibrated" \
      --override "loss.type=memory_self_calibrated" \
      --override "model.fast_encoder.input_frames=remaining" \
      --override "memory_self_calibrated.actual_trace_start_epoch=${ACTUAL_TRACE_START_EPOCH}" \
      --override "memory_self_calibrated.hard_mining_start_epoch=${HARD_MINING_START_EPOCH}" \
      --override "memory_self_calibrated.raw_positive_weight=1.0" \
      --override "memory_self_calibrated.planned_positive_weight=0.5" \
      --override "memory_self_calibrated.missed_positive_weight=1.25" \
      --override "memory_self_calibrated.hard_negative_weight=1.10" \
      --override "memory_self_calibrated.trust_momentum=0.9" \
      --override "memory_self_calibrated.edge_momentum=0.9" \
      --override "memory_self_calibrated.edge_slots=64" \
      --override "memory_self_calibrated.planned_positive_topk=5" \
      --override "memory_self_calibrated.missed_positive_topk=5"
  done

  echo "$(timestamp) | rf_clath_hmdb_memory_self_calibrated done"
}

log_file="${LOG_ROOT}/rf_clath_hmdb_memory_self_calibrated_cuda${GPU}_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | RF-CLaTH HMDB memory-self-calibrated run start, bits=${BITS}, gpu=${GPU}, log=${log_file}"
run_hmdb >> "$log_file" 2>&1
echo "$(timestamp) | RF-CLaTH HMDB memory-self-calibrated run done"
