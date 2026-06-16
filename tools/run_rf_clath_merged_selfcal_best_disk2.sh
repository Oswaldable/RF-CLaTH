#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

DATASET_KEY="${1:?usage: $0 <hmdb|ucf|act|activitynet|fcv> [gpu]}"
GPU="${2:-0}"
BITS="${BITS:-16 32 64}"
EPOCHS="${EPOCHS:-150}"
CUT="${CUT:-35}"
OMEGA_Z="${OMEGA_Z:-0.30}"
LAMBDA_MERGED="${LAMBDA_MERGED:-0.8}"
VIEW_POSITIVE_WEIGHT="${VIEW_POSITIVE_WEIGHT:-1.0}"
NEIGHBOR_POSITIVE_WEIGHT="${NEIGHBOR_POSITIVE_WEIGHT:-1.0}"
MAX_POSITIVE_WEIGHT="${MAX_POSITIVE_WEIGHT:-2.0}"
LAMBDA_MEMORY_NEIGHBOR="${LAMBDA_MEMORY_NEIGHBOR:-0.30}"
MEMORY_POSITIVES_PER_ANCHOR="${MEMORY_POSITIVES_PER_ANCHOR:-3}"
NEIGHBOR_TEMPERATURE="${NEIGHBOR_TEMPERATURE:-0.2}"
MEMORY_TEMPERATURE="${MEMORY_TEMPERATURE:-0.2}"
TOP_R="${TOP_R:-20}"
FEEDBACK_RAMP_EPOCHS="${FEEDBACK_RAMP_EPOCHS:-1}"
RAW_TRUST_TOPK="${RAW_TRUST_TOPK:-5}"
RUN_SUFFIX="${RUN_SUFFIX:-best_memw030_trust5_view10_temp02_remaining_fast_disk2}"

case "$DATASET_KEY" in
  hmdb)
    CONFIG=configs/rf_clath_hmdb.yaml
    DATASET_ARG=hmdb
    TAG=hmdb
    PROJECT_DATASET=HMDB
    ;;
  ucf)
    CONFIG=configs/rf_clath_ucf.yaml
    DATASET_ARG=ucf
    TAG=ucf
    PROJECT_DATASET=UCF
    ;;
  act|activitynet)
    CONFIG=configs/rf_clath_activitynet.yaml
    DATASET_ARG=activitynet
    TAG=activitynet
    PROJECT_DATASET=ActivityNet
    ;;
  fcv|fcvid)
    CONFIG=configs/rf_clath_fcv.yaml
    DATASET_ARG=fcv
    TAG=fcv
    PROJECT_DATASET=FCVID
    ;;
  *)
    echo "Unsupported dataset key: ${DATASET_KEY}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_ROOT}/rf_clath_${TAG}_merged_selfcal_${RUN_SUFFIX}"
WARMUP=$((CUT - 1))
if [ "$WARMUP" -lt 0 ]; then
  WARMUP=0
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

run_dataset() {
  cd "$PROJECT_ROOT"

  for bits in $BITS; do
    echo "$(timestamp) | merged_selfcal_best dataset=${TAG} ${bits}-bit start on cuda${GPU}, cut=${CUT}, memw=${LAMBDA_MEMORY_NEIGHBOR}, mem_pos=${MEMORY_POSITIVES_PER_ANCHOR}, raw_trust_topk=${RAW_TRUST_TOPK}, view=${VIEW_POSITIVE_WEIGHT}, temp=${NEIGHBOR_TEMPERATURE}, suffix=${RUN_SUFFIX}"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" train.py \
      --config "$CONFIG" \
      --dataset "$DATASET_ARG" \
      --device cuda \
      --output-dir "$OUTPUT_DIR/cut${CUT}" \
      --epochs "$EPOCHS" \
      --hash-bits "$bits" \
      --override "project.name=RF-CLaTH-${PROJECT_DATASET}-MergedSelfCal-${RUN_SUFFIX}-Cut${CUT}" \
      --override "training.objective=merged_semantic_self_calibrated" \
      --override "train.objective=merged_semantic_self_calibrated" \
      --override "loss.type=merged_semantic_self_calibrated" \
      --override "model.fast_encoder.input_frames=remaining" \
      --override "loss.semantic.lambda_merged=${LAMBDA_MERGED}" \
      --override "loss.semantic.view_positive_weight=${VIEW_POSITIVE_WEIGHT}" \
      --override "loss.semantic.neighbor_positive_weight=${NEIGHBOR_POSITIVE_WEIGHT}" \
      --override "loss.semantic.max_positive_weight=${MAX_POSITIVE_WEIGHT}" \
      --override "loss.semantic.lambda_memory_neighbor=${LAMBDA_MEMORY_NEIGHBOR}" \
      --override "loss.memory_neighbor.positives_per_anchor=${MEMORY_POSITIVES_PER_ANCHOR}" \
      --override "loss.neighbor_temperature=${NEIGHBOR_TEMPERATURE}" \
      --override "loss.memory_neighbor.temperature=${MEMORY_TEMPERATURE}" \
      --override "planner.omega_s=0.45" \
      --override "planner.omega_t=0.25" \
      --override "planner.omega_z=${OMEGA_Z}" \
      --override "planner.warmup.epochs=${WARMUP}" \
      --override "planner.warmup.omega_s=0.65" \
      --override "planner.warmup.omega_t=0.35" \
      --override "planner.warmup.omega_z=0.0" \
      --override "retrieval_environment.use_actual_trace=true" \
      --override "retrieval_environment.top_r=${TOP_R}" \
      --override "feedback.eta_missed_start=0.0" \
      --override "feedback.eta_false_start=0.0" \
      --override "feedback.eta_missed_final=1.0" \
      --override "feedback.eta_false_final=1.0" \
      --override "feedback.ramp_epochs=${FEEDBACK_RAMP_EPOCHS}" \
      --override "memory_self_calibrated.actual_trace_start_epoch=${CUT}" \
      --override "memory_self_calibrated.hard_mining_start_epoch=${CUT}" \
      --override "memory_self_calibrated.raw_trust_topk=${RAW_TRUST_TOPK}" \
      --override "train.eval_interval=5" \
      --override "train.save_interval=5"
  done

  echo "$(timestamp) | merged_selfcal_best dataset=${TAG} done"
}

log_file="${LOG_ROOT}/rf_clath_${TAG}_merged_selfcal_${RUN_SUFFIX}_cuda${GPU}_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | RF-CLaTH merged-selfcal best run start, dataset=${TAG}, bits=${BITS}, gpu=${GPU}, cut=${CUT}, memw=${LAMBDA_MEMORY_NEIGHBOR}, mem_pos=${MEMORY_POSITIVES_PER_ANCHOR}, raw_trust_topk=${RAW_TRUST_TOPK}, log=${log_file}"
run_dataset >> "$log_file" 2>&1
echo "$(timestamp) | RF-CLaTH merged-selfcal best run done, dataset=${TAG}"
