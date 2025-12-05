#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-game}
GPU_ID=${2:-0}
EPOCH=${3:-200}

echo "Running SASRec baseline for dataset: $DATASET on GPU: $GPU_ID"

# Define paths
DLLM2REC_DIR="DLLM2Rec"

cd "$DLLM2REC_DIR"

# Run training
# Set weights to 0 to disable distillation and regularization for pure SASRec baseline
SECONDS=0
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --data "$DATASET" \
    --model_name "SASRec" \
    --epoch "$EPOCH" \
    --alpha 0 \
    --ed_weight 0 \
    --lam 0 \
    --cuda 0

duration=$SECONDS
echo "SASRec baseline training completed in $(($duration / 60)) minutes and $(($duration % 60)) seconds"
