#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-movie}
GPU_ID=${2:-0}
SEED=${3:-0}
SAMPLE=${4:-1024}

echo "Running BIGRec training for dataset: $DATASET"

# Define paths
BIGREC_DIR="BIGRec"
OUTPUT_DIR="./model/$DATASET/${SEED}_${SAMPLE}"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

cd "$BIGREC_DIR"

# Run training
# Note: Adjust arguments as needed based on README and requirements
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --base_model "Qwen/Qwen2-0.5B" \
    --train_data_path "[\"./data/$DATASET/train.json\"]" \
    --val_data_path "[\"./data/$DATASET/valid_5000.json\"]" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 50 \
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
