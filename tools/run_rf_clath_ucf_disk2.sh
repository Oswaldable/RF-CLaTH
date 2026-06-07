#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/disk2/yql/RF-CLaTH
PYTHON_BIN=/mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
OUTPUT_ROOT=/mnt/disk2/yql/RF-CLaTH_outputs
LOG_ROOT=/mnt/disk2/yql/RF-CLaTH_run_logs

GPU="${1:-1}"
BITS="${BITS:-16 32 64 128}"

CONFIG=configs/rf_clath_ucf.yaml
DATASET=s5vh_ucf
OUTPUT_DIR="${OUTPUT_ROOT}/rf_clath_ucf_disk2"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

run_ucf() {
  cd "$PROJECT_ROOT"

  for bits in $BITS; do
    echo "$(timestamp) | rf_clath_ucf ${bits}-bit start on cuda${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" train.py \
      --config "$CONFIG" \
      --dataset "$DATASET" \
      --device cuda \
      --output-dir "$OUTPUT_DIR" \
      --epochs 150 \
      --hash-bits "$bits" \
      --override "project.name=RF-CLaTH-UCF-RePartition"
  done

  echo "$(timestamp) | rf_clath_ucf done"
}

log_file="${LOG_ROOT}/rf_clath_ucf_disk2_$(date +%Y%m%d_%H%M%S).queue.log"
echo "$(timestamp) | RF-CLaTH UCF run start, bits=${BITS}, gpu=${GPU}, log=${log_file}"
run_ucf >> "$log_file" 2>&1
echo "$(timestamp) | RF-CLaTH UCF run done"
