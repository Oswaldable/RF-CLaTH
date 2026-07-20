#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

GPU="${1:?usage: $0 <gpu> <h0|h1>}"
CASE_ID="${2:?usage: $0 <gpu> <h0|h1>}"

CONFIG=configs/rf_clath_ucf.yaml
DATASET=s5vh_ucf
BITS=32
EPOCHS=15

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

overrides=(
  "project.seed=3346"
  "agentic.policy.use_routed_similarity=true"
  "agentic.policy.update_feedback_graph=false"
  "agentic.stop_policy.enabled=false"
  "agentic_contrastive.hard_mining_start_epoch=999"
  "agentic_contrastive.memory_candidate_start_epoch=6"
  "agentic_contrastive.memory_candidate_ramp_epochs=5"
  "agentic_contrastive.memory_source_ramp_epochs=5"
  "agentic_contrastive.memory_source_ramp_apply_to_arf=true"
  "agentic_contrastive.memory_candidate_strategy=all"
  "agentic_contrastive.memory_candidate_topk=0"
  "feedback.eta_missed_start=0.0"
  "feedback.eta_false_start=0.0"
  "feedback.eta_missed_final=0.0"
  "feedback.eta_false_final=0.0"
  "train.eval_interval=5"
  "train.save_interval=5"
)

case "$CASE_ID" in
  h0)
    case_name=h0_trace_off_control_15ep
    project_name=RF-CLaTH-UCF-H0-TraceOff-Control-15Ep
    overrides+=(
      "agentic_contrastive.actual_trace_start_epoch=999"
      "retrieval_environment.use_actual_trace=false"
    )
    ;;
  h1)
    case_name=h1_trace_only_from_epoch7_15ep
    project_name=RF-CLaTH-UCF-H1-TraceOnly-Epoch7-15Ep
    overrides+=(
      "agentic_contrastive.actual_trace_start_epoch=7"
      "retrieval_environment.use_actual_trace=true"
    )
    ;;
  *)
    echo "Unknown case=${CASE_ID}; expected h0 or h1." >&2
    exit 2
    ;;
esac

output_dir="${OUTPUT_ROOT}/rf_clath_ucf_h0_h1_actual_trace/${case_name}"
command=(
  "$PYTHON_BIN" train.py
  --config "$CONFIG"
  --dataset "$DATASET"
  --device cuda
  --output-dir "$output_dir"
  --epochs "$EPOCHS"
  --hash-bits "$BITS"
  --override "project.name=${project_name}"
)
for override in "${overrides[@]}"; do
  command+=(--override "$override")
done

log_file="${LOG_ROOT}/rf_clath_ucf_${CASE_ID}_actual_trace_cuda${GPU}_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | case=${CASE_ID} name=${case_name} bits=${BITS} epochs=${EPOCHS} gpu=${GPU} log=${log_file}"
cd "$PROJECT_ROOT" || exit 1
if CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "${command[@]}" >> "$log_file" 2>&1; then
  echo "$(timestamp) | case=${CASE_ID} completed on cuda${GPU}"
  exit 0
else
  status=$?
  echo "$(timestamp) | case=${CASE_ID} failed status=${status} on cuda${GPU}" >&2
  exit "$status"
fi
