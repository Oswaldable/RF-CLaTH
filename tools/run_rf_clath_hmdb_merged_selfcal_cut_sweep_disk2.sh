#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

GPU="${1:-2}"
BITS="${BITS:-16}"
EPOCHS="${EPOCHS:-70}"
CUTS="${CUTS:-20 25 30 35}"
OMEGA_Z="${OMEGA_Z:-0.30}"

CONFIG=configs/rf_clath_hmdb.yaml
DATASET=hmdb
OUTPUT_DIR="${OUTPUT_ROOT}/rf_clath_hmdb_merged_selfcal_omegaz_sweep_remaining_fast_disk2"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

run_hmdb() {
  cd "$PROJECT_ROOT"

  for cut in $CUTS; do
    warmup=$((cut - 1))
    if [ "$warmup" -lt 0 ]; then
      warmup=0
    fi
    for bits in $BITS; do
      echo "$(timestamp) | merged_selfcal_omegaz cut=${cut} ${bits}-bit start on cuda${GPU}"
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" train.py \
        --config "$CONFIG" \
        --dataset "$DATASET" \
        --device cuda \
        --output-dir "$OUTPUT_DIR/cut${cut}" \
        --epochs "$EPOCHS" \
        --hash-bits "$bits" \
        --override "project.name=RF-CLaTH-HMDB-MergedSelfCal-OmegaZ-Cut${cut}" \
        --override "training.objective=merged_semantic_self_calibrated" \
        --override "train.objective=merged_semantic_self_calibrated" \
        --override "loss.type=merged_semantic_self_calibrated" \
        --override "model.fast_encoder.input_frames=remaining" \
        --override "loss.semantic.lambda_merged=0.8" \
        --override "loss.semantic.view_positive_weight=1.2" \
        --override "loss.semantic.neighbor_positive_weight=1.0" \
        --override "loss.semantic.max_positive_weight=2.0" \
        --override "loss.semantic.lambda_memory_neighbor=0.04" \
        --override "loss.neighbor_temperature=0.2" \
        --override "loss.memory_neighbor.temperature=0.2" \
        --override "planner.omega_s=0.45" \
        --override "planner.omega_t=0.25" \
        --override "planner.omega_z=${OMEGA_Z}" \
        --override "planner.warmup.epochs=${warmup}" \
        --override "planner.warmup.omega_s=0.65" \
        --override "planner.warmup.omega_t=0.35" \
        --override "planner.warmup.omega_z=0.0" \
        --override "retrieval_environment.use_actual_trace=true" \
        --override "retrieval_environment.top_r=20" \
        --override "feedback.eta_missed_start=0.0" \
        --override "feedback.eta_false_start=0.0" \
        --override "feedback.eta_missed_final=1.0" \
        --override "feedback.eta_false_final=1.0" \
        --override "feedback.ramp_epochs=1" \
        --override "memory_self_calibrated.actual_trace_start_epoch=${cut}" \
        --override "memory_self_calibrated.hard_mining_start_epoch=${cut}" \
        --override "train.eval_interval=5" \
        --override "train.save_interval=5"
    done
  done

  echo "$(timestamp) | merged_selfcal_omegaz sweep done"
}

log_file="${LOG_ROOT}/rf_clath_hmdb_merged_selfcal_omegaz_sweep_cuda${GPU}_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | RF-CLaTH HMDB merged-selfcal omega_z sweep start, cuts=${CUTS}, bits=${BITS}, gpu=${GPU}, log=${log_file}"
run_hmdb >> "$log_file" 2>&1
echo "$(timestamp) | RF-CLaTH HMDB merged-selfcal omega_z sweep done"
