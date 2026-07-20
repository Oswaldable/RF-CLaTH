#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

GPU="${1:?usage: $0 <gpu>}"
CONFIG=configs/rf_clath_ucf.yaml
DATASET=s5vh_ucf
BITS=32
EPOCHS=15
CASE_NAME=i0_trace_depth_score_15ep
PROJECT_NAME=RF-CLaTH-UCF-I0-TraceDepthScore-15Ep

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

overrides=(
  "project.seed=3346"
  "agentic.policy.use_routed_similarity=true"
  "agentic.policy.update_feedback_graph=false"
  "agentic.policy.budget_min=20"
  "agentic.policy.budget_max=20"
  "agentic.stop_policy.enabled=false"
  "agentic_contrastive.actual_trace_start_epoch=7"
  "agentic_contrastive.hard_mining_start_epoch=999"
  "agentic_contrastive.memory_candidate_start_epoch=6"
  "agentic_contrastive.memory_candidate_ramp_epochs=5"
  "agentic_contrastive.memory_source_ramp_epochs=5"
  "agentic_contrastive.memory_source_ramp_apply_to_arf=true"
  "agentic_contrastive.memory_candidate_strategy=all"
  "agentic_contrastive.memory_candidate_topk=0"
  "retrieval_environment.use_actual_trace=true"
  "feedback.eta_missed_start=0.0"
  "feedback.eta_false_start=0.0"
  "feedback.eta_missed_final=0.0"
  "feedback.eta_false_final=0.0"
  "train.eval_interval=5"
  "train.save_interval=5"
)

output_dir="${OUTPUT_ROOT}/rf_clath_ucf_i0_trace_depth_score/${CASE_NAME}"
command=(
  "$PYTHON_BIN" train.py
  --config "$CONFIG"
  --dataset "$DATASET"
  --device cuda
  --output-dir "$output_dir"
  --epochs "$EPOCHS"
  --hash-bits "$BITS"
  --override "project.name=${PROJECT_NAME}"
)
for override in "${overrides[@]}"; do
  command+=(--override "$override")
done

log_file="${LOG_ROOT}/rf_clath_ucf_i0_trace_depth_score_cuda${GPU}_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | case=i0 name=${CASE_NAME} bits=${BITS} epochs=${EPOCHS} gpu=${GPU} log=${log_file}"
cd "$PROJECT_ROOT" || exit 1
if CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "${command[@]}" >> "$log_file" 2>&1; then
  echo "$(timestamp) | case=i0 completed on cuda${GPU}"
  exit 0
else
  status=$?
  echo "$(timestamp) | case=i0 failed status=${status} on cuda${GPU}" >&2
  exit "$status"
fi
