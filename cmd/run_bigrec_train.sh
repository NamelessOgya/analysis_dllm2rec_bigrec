#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-movie}
GPU_ID=${2:-0}
SEED=${3:-0}
SAMPLE=${4:-1024}
BATCH_SIZE=${5:-128}
MICRO_BATCH_SIZE=${6:-4}
BASE_MODEL=${7:-"Qwen/Qwen2-0.5B"}
NUM_EPOCHS=${8:-50}

echo "Running BIGRec training for dataset: $DATASET"

# Sanitize model name for directory usage (replace / with _)
SAFE_MODEL_NAME=$(echo "$BASE_MODEL" | tr '/' '_')

# Define paths
BIGREC_DIR="BIGRec"
# Use absolute path for output directory to avoid issues when changing directory
OUTPUT_DIR="$(pwd)/BIGRec/model/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

cd "$BIGREC_DIR"

# Run training
# Note: Adjust arguments as needed based on README and requirements
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --base_model "$BASE_MODEL" \
    --train_data_path "[\"./data/$DATASET/train.json\"]" \
    --val_data_path "[\"./data/$DATASET/valid_5000.json\"]" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --seed $SEED \
    --sample $SAMPLE

echo "BIGRec training completed."
