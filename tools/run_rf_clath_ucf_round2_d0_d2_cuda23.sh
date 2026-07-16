#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

GPU="${1:?usage: $0 <gpu> <case> [case ...]}"
shift
if [ "$#" -eq 0 ]; then
  echo "At least one case is required." >&2
  exit 2
fi

CONFIG=configs/rf_clath_ucf.yaml
DATASET=s5vh_ucf
BITS=32
EPOCHS="${EPOCHS:-10}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

run_case() {
  local case_id="$1"
  local case_name
  local project_name
  local -a overrides

  overrides=(
    "project.seed=3346"
    "agentic.policy.use_routed_similarity=true"
    "agentic.policy.update_feedback_graph=false"
    "agentic.stop_policy.enabled=false"
    "agentic_contrastive.actual_trace_start_epoch=999"
    "agentic_contrastive.hard_mining_start_epoch=999"
    "agentic_contrastive.memory_candidate_ramp_epochs=0"
    "agentic_contrastive.memory_source_ramp_epochs=0"
    "agentic_contrastive.memory_source_ramp_apply_to_arf=true"
    "retrieval_environment.use_actual_trace=false"
    "feedback.eta_missed_start=0.0"
    "feedback.eta_false_start=0.0"
    "feedback.eta_missed_final=0.0"
    "feedback.eta_false_final=0.0"
    "train.eval_interval=5"
    "train.save_interval=5"
  )

  case "$case_id" in
    d0)
      case_name=d0_route_memory_from_epoch1
      project_name=RF-CLaTH-UCF-Round2-D0-RouteMemoryFromEpoch1
      overrides+=("agentic_contrastive.memory_candidate_start_epoch=1")
      ;;
    d1)
      case_name=d1_routed_memory_hard_warmup
      project_name=RF-CLaTH-UCF-Round2-D1-RoutedMemoryHardWarmup
      overrides+=("agentic_contrastive.memory_candidate_start_epoch=6")
      ;;
    d2)
      case_name=d2_routed_memory_linear_ramp
      project_name=RF-CLaTH-UCF-Round2-D2-RoutedMemoryLinearRamp
      overrides+=(
        "agentic_contrastive.memory_candidate_start_epoch=6"
        "agentic_contrastive.memory_candidate_ramp_epochs=5"
        "agentic_contrastive.memory_source_ramp_epochs=5"
      )
      ;;
    *)
      echo "$(timestamp) | unknown case=${case_id}" >&2
      return 2
      ;;
  esac

  local output_dir="${OUTPUT_ROOT}/rf_clath_ucf_round2/${case_name}"
  local -a command
  local override
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

  echo "$(timestamp) | case=${case_id} name=${case_name} bits=${BITS} epochs=${EPOCHS} start on cuda${GPU}"
  if CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "${command[@]}"; then
    echo "$(timestamp) | case=${case_id} name=${case_name} completed on cuda${GPU}"
    return 0
  else
    local status=$?
    echo "$(timestamp) | case=${case_id} name=${case_name} failed status=${status} on cuda${GPU}" >&2
    return "$status"
  fi
}

run_queue() {
  local case_id
  cd "$PROJECT_ROOT" || return 1
  for case_id in "$@"; do
    run_case "$case_id" || true
  done
  echo "$(timestamp) | queue completed on cuda${GPU}: $*"
}

queue_tag="$(IFS=-; echo "$*")"
log_file="${LOG_ROOT}/rf_clath_ucf_round2_cuda${GPU}_${queue_tag}_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | round2 queue start gpu=${GPU} cases=$* log=${log_file}"
run_queue "$@" >> "$log_file" 2>&1
echo "$(timestamp) | round2 queue launcher done gpu=${GPU} cases=$*"
