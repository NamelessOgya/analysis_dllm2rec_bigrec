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
# Run training
SECONDS=0
# Note: Adjust arguments as needed based on README and requirements

# Calculate number of GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPU_ID"
NUM_GPUS=${#GPU_ARRAY[@]}

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Detected $NUM_GPUS GPUs. Using torchrun for distributed training."
    CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 train.py \
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
        --sample $SAMPLE \
        $( [ -n "$PROMPT_FILE" ] && echo "--prompt_file $PROMPT_FILE" )
else
    echo "Using single GPU training."
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
        --sample $SAMPLE \
        $( [ -n "$PROMPT_FILE" ] && echo "--prompt_file $PROMPT_FILE" )
fi

duration=$SECONDS
duration_min=$(($duration / 60))
echo "Finetuning time: $duration_min minutes"

# Save execution time to JSON
python -c "import json; import os; 
data = {'finetuning_time_minutes': $duration_min}; 
with open(os.path.join('$OUTPUT_DIR', 'execution_time.json'), 'w') as f: json.dump(data, f, indent=4)"

echo "BIGRec training completed."
