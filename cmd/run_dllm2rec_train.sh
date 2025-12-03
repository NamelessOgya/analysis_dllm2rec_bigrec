#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-game}
MODEL_NAME=${2:-SASRec}
GPU_ID=${3:-0}

echo "Running DLLM2Rec training for dataset: $DATASET with model: $MODEL_NAME"

# Define paths
DLLM2REC_DIR="DLLM2Rec"

cd "$DLLM2REC_DIR"

# Run training
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --data "$DATASET" \
    --model_name "$MODEL_NAME" \
    --alpha 0.5 \
    --ed_weight 0.3 \
    --lam 0.7 \
    --cuda 0

echo "DLLM2Rec training completed."
