#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

GPU="${1:?usage: $0 <gpu> <g0|g1>}"
CASE_ID="${2:?usage: $0 <gpu> <g0|g1>}"

CONFIG=configs/rf_clath_ucf.yaml
DATASET=s5vh_ucf
BITS=32
EPOCHS=30

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

overrides=(
  "project.seed=3346"
  "agentic.policy.use_routed_similarity=true"
  "agentic.policy.update_feedback_graph=false"
  "agentic.stop_policy.enabled=false"
  "agentic_contrastive.actual_trace_start_epoch=999"
  "agentic_contrastive.hard_mining_start_epoch=999"
  "agentic_contrastive.memory_candidate_start_epoch=6"
  "agentic_contrastive.memory_candidate_ramp_epochs=5"
  "agentic_contrastive.memory_source_ramp_epochs=5"
  "agentic_contrastive.memory_source_ramp_apply_to_arf=true"
  "retrieval_environment.use_actual_trace=false"
  "feedback.eta_missed_start=0.0"
  "feedback.eta_false_start=0.0"
  "feedback.eta_missed_final=0.0"
  "feedback.eta_false_final=0.0"
  "train.eval_interval=5"
  "train.save_interval=5"
)

case "$CASE_ID" in
  g0)
    case_name=g0_routed_memory_supervised_only_30ep
    project_name=RF-CLaTH-UCF-G0-MemorySupervisedOnly-30Ep
    memory_candidate_strategy=supervised_only
    memory_candidate_topk=0
    ;;
  g1)
    case_name=g1_routed_memory_uniform1024_30ep
    project_name=RF-CLaTH-UCF-G1-MemoryUniform1024-30Ep
    memory_candidate_strategy=uniform
    memory_candidate_topk=1024
    ;;
  *)
    echo "Unknown case=${CASE_ID}; expected g0 or g1." >&2
    exit 2
    ;;
esac

overrides+=(
  "agentic_contrastive.memory_candidate_strategy=${memory_candidate_strategy}"
  "agentic_contrastive.memory_candidate_topk=${memory_candidate_topk}"
)
output_dir="${OUTPUT_ROOT}/rf_clath_ucf_g0_g1_memory_pool/${case_name}"
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

log_file="${LOG_ROOT}/rf_clath_ucf_${CASE_ID}_memory_pool_cuda${GPU}_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | case=${CASE_ID} name=${case_name} strategy=${memory_candidate_strategy} memory_topk=${memory_candidate_topk} bits=${BITS} epochs=${EPOCHS} gpu=${GPU} log=${log_file}"
cd "$PROJECT_ROOT" || exit 1
if CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "${command[@]}" >> "$log_file" 2>&1; then
  echo "$(timestamp) | case=${CASE_ID} completed on cuda${GPU}"
  exit 0
else
  status=$?
  echo "$(timestamp) | case=${CASE_ID} failed status=${status} on cuda${GPU}" >&2
  exit "$status"
fi
