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
SECONDS=0
python main.py \
    --data "$DATASET" \
    --model_name "$MODEL_NAME" \
    --alpha 0.5 \
    --ed_weight 0.3 \
    --lam 0.7 \
    --cuda "$GPU_ID"

duration=$SECONDS
duration_min=$(($duration / 60))
echo "Distillation process time: $duration_min minutes"

# Create output directory if not exists (DLLM2Rec doesn't seem to have a standard output dir structure in this script)
OUTPUT_DIR="output/$DATASET"
mkdir -p "$OUTPUT_DIR"

# Save execution time to JSON
python -c "import json; import os; 
data = {'distillation_time_minutes': $duration_min}; 
with open(os.path.join('$OUTPUT_DIR', 'execution_time.json'), 'w') as f: json.dump(data, f, indent=4)"

echo "DLLM2Rec training completed."
