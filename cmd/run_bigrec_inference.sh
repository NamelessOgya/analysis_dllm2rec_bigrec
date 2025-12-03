#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-movie}
GPU_ID=${2:-0}
BASE_MODEL=${3:-"Qwen/Qwen2-0.5B"}
SEED=${4:-0}
SAMPLE=${5:-1024}

echo "Running BIGRec inference for dataset: $DATASET"

# Define paths
BIGREC_DIR="BIGRec"
RESULT_DIR="./${DATASET}_result"
TEST_DATA_PATH="./data/$DATASET/test_5000.json"
RESULT_JSON_PATH="$RESULT_DIR/${DATASET}.json"

# Sanitize model name for directory usage (replace / with _)
SAFE_MODEL_NAME=$(echo "$BASE_MODEL" | tr '/' '_')

# Construct LoRA weights path
# Path format: ./model/<dataset>/<safe_model_name>/<seed>_<sample>
LORA_WEIGHTS="./model/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}"

# Ensure result directory exists
mkdir -p "$BIGREC_DIR/$RESULT_DIR"

cd "$BIGREC_DIR"

# Check if LoRA weights exist
if [ ! -d "$LORA_WEIGHTS" ]; then
    echo "Error: LoRA weights not found at $LORA_WEIGHTS"
    echo "Please check your arguments (dataset, model, seed, sample) or run training first."
    exit 1
fi

echo "Using LoRA weights from: $LORA_WEIGHTS"

# Run inference
CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
    --base_model "$BASE_MODEL" \
    --lora_weights "$LORA_WEIGHTS" \
    --test_data_path "$TEST_DATA_PATH" \
    --result_json_data "$RESULT_JSON_PATH"

echo "BIGRec inference completed."
