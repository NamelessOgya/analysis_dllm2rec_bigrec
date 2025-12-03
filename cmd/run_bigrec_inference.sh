#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-movie}
GPU_ID=${2:-0}

echo "Running BIGRec inference for dataset: $DATASET"

# Define paths
BIGREC_DIR="BIGRec"
RESULT_DIR="./${DATASET}_result"
TEST_DATA_PATH="./data/$DATASET/test_5000.json"
RESULT_JSON_PATH="$RESULT_DIR/${DATASET}.json"

# Ensure result directory exists
mkdir -p "$BIGREC_DIR/$RESULT_DIR"

cd "$BIGREC_DIR"

# Run inference
# Note: You need to specify where the LoRA weights are. 
# Assuming they are in model/{dataset}/0_1024 (default from train script)
LORA_WEIGHTS="./model/$DATASET/0_1024"

CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
    --base_model "Qwen/Qwen2-0.5B" \
    --lora_weights "$LORA_WEIGHTS" \
    --test_data_path "$TEST_DATA_PATH" \
    --result_json_data "$RESULT_JSON_PATH"

echo "BIGRec inference completed."
