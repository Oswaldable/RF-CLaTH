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
    "agentic.policy.update_feedback_graph=false"
    "agentic.stop_policy.enabled=false"
    "agentic_contrastive.actual_trace_start_epoch=999"
    "agentic_contrastive.hard_mining_start_epoch=999"
    "retrieval_environment.use_actual_trace=false"
    "feedback.eta_missed_start=0.0"
    "feedback.eta_false_start=0.0"
    "feedback.eta_missed_final=0.0"
    "feedback.eta_false_final=0.0"
    "train.eval_interval=5"
    "train.save_interval=5"
  )

  case "$case_id" in
    c0)
      case_name=c0_legacy_control
      project_name=RF-CLaTH-UCF-Round1-C0-LegacyControl
      overrides+=(
        "training.objective=merged_semantic_self_calibrated"
        "train.objective=merged_semantic_self_calibrated"
        "loss.type=merged_semantic_self_calibrated"
        "agentic.enabled=false"
        "agentic.policy.use_subcode_concat=false"
        "model.fast_encoder.input_frames=remaining"
        "loss.semantic.lambda_merged=0.8"
        "loss.semantic.view_positive_weight=1.0"
        "loss.semantic.neighbor_positive_weight=1.0"
        "loss.semantic.max_positive_weight=2.0"
        "loss.semantic.lambda_memory_neighbor=0.30"
        "loss.memory_neighbor.positives_per_anchor=3"
        "memory_self_calibrated.actual_trace_start_epoch=999"
        "memory_self_calibrated.hard_mining_start_epoch=999"
        "memory_self_calibrated.raw_trust_topk=5"
      )
      ;;
    c1)
      case_name=c1_no_routed_similarity
      project_name=RF-CLaTH-UCF-Round1-C1-NoRoutedSimilarity
      overrides+=(
        "agentic.policy.use_routed_similarity=false"
        "agentic_contrastive.memory_candidate_start_epoch=1"
      )
      ;;
    c2)
      case_name=c2_no_memory_candidates
      project_name=RF-CLaTH-UCF-Round1-C2-NoMemoryCandidates
      overrides+=(
        "agentic.policy.use_routed_similarity=true"
        "agentic_contrastive.memory_candidate_start_epoch=11"
      )
      ;;
    c3)
      case_name=c3_no_route_memory_warmup
      project_name=RF-CLaTH-UCF-Round1-C3-NoRouteMemoryWarmup
      overrides+=(
        "agentic.policy.use_routed_similarity=false"
        "agentic_contrastive.memory_candidate_start_epoch=6"
      )
      ;;
    *)
      echo "$(timestamp) | unknown case=${case_id}" >&2
      return 2
      ;;
  esac

  local output_dir="${OUTPUT_ROOT}/rf_clath_ucf_round1/${case_name}"
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
log_file="${LOG_ROOT}/rf_clath_ucf_round1_cuda${GPU}_${queue_tag}_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | round1 queue start gpu=${GPU} cases=$* log=${log_file}"
run_queue "$@" >> "$log_file" 2>&1
echo "$(timestamp) | round1 queue launcher done gpu=${GPU} cases=$*"
