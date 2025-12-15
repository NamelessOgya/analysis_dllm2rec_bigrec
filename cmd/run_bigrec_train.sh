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
PROMPT_FILE=${9:-""}
TRAIN_DATA_FILE=${10:-"train.json"}
MODEL_SUFFIX=${11:-""}

echo "Running BIGRec training for dataset: $DATASET"
echo "  - Train Data: $TRAIN_DATA_FILE"
echo "  - Suffix: $MODEL_SUFFIX"

# Sanitize model name for directory usage (replace / with _)
SAFE_MODEL_NAME=$(echo "$BASE_MODEL" | tr '/' '_')

# Define paths
BIGREC_DIR="BIGRec"
# Use absolute path for output directory to avoid issues when changing directory
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: OUTPUT_DIR environment variable is required."
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

cd "$BIGREC_DIR"

# Run training
# Run training
# Start timing
START_TIME=$(python3 -c 'import time; print(time.time())')

# Resolve Training Data Path
# Resolve Training Data Path
# Check if path is absolute, starts with ./, or starts with experiments/
if [[ "$TRAIN_DATA_FILE" == /* ]] || [[ "$TRAIN_DATA_FILE" == ./* ]] || [[ "$TRAIN_DATA_FILE" == experiments/* ]]; then
    FULL_TRAIN_PATH="$TRAIN_DATA_FILE"
    # If path is relative (starts with experiments/ or ./), we need to make sure it resolves correctly from BIGRec dir.
    # The script cd's into BIGRec dir (line 37).
    # If the path is relative to repo root (like experiments/...), then from BIGRec dir it should be ../experiments/...
    if [[ "$TRAIN_DATA_FILE" == experiments/* ]]; then
         FULL_TRAIN_PATH="../$TRAIN_DATA_FILE"
    fi
else
    FULL_TRAIN_PATH="./data/$DATASET/$TRAIN_DATA_FILE"
fi

# Calculate number of GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPU_ID"
NUM_GPUS=${#GPU_ARRAY[@]}

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Detected $NUM_GPUS GPUs. Using torchrun for distributed training."
    CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 train.py \
        --base_model "$BASE_MODEL" \
        --train_data_path "[\"$FULL_TRAIN_PATH\"]" \
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
        --sample $SAMPLE \
        $( [ -n "$PROMPT_FILE" ] && echo "--prompt_file $PROMPT_FILE" )
else
    echo "Using single GPU training."
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        --base_model "$BASE_MODEL" \
        --train_data_path "[\"$FULL_TRAIN_PATH\"]" \
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
        --sample $SAMPLE \
        $( [ -n "$PROMPT_FILE" ] && echo "--prompt_file $PROMPT_FILE" )
fi

# End timing
END_TIME=$(python3 -c 'import time; print(time.time())')
ELAPSED=$(python3 -c "print($END_TIME - $START_TIME)")
ELAPSED_MIN=$(python3 -c "print($ELAPSED / 60)")

echo "Finetuning time: $ELAPSED seconds ($ELAPSED_MIN minutes)"

# Save execution time to JSON
python -c "import json; import os; 
data = {'finetuning_time_minutes': $ELAPSED_MIN, 'finetuning_time_seconds': $ELAPSED}; 
with open(os.path.join('$OUTPUT_DIR', 'execution_time.json'), 'w') as f: json.dump(data, f, indent=4)"

echo "BIGRec training completed."
