#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-game}
MODEL_NAME=${2:-SASRec}
GPU_ID=${3:-0}
ED_WEIGHT=${4:-0.3}
LAM=${5:-0.7}
BIGREC_BASE_MODEL=${6:-""}
BIGREC_SEED=${7:-0}
BIGREC_SAMPLE=${8:-1024}

echo "Running DLLM2Rec training for dataset: $DATASET with model: $MODEL_NAME"
echo "Params: ed_weight=$ED_WEIGHT, lam=$LAM"

EXTRA_ARGS=""
if [ -n "$BIGREC_BASE_MODEL" ]; then
    echo "Using direct paths for BIGRec results (skipping tocf copy)..."
    SAFE_MODEL_NAME=$(echo "$BIGREC_BASE_MODEL" | tr '/' '_')
    ROOT_DIR=$(pwd)
    
    # Construct paths
    EMBEDDING_PATH="$ROOT_DIR/BIGRec/data/$DATASET/model_embeddings/${SAFE_MODEL_NAME}.pt"
    RESULT_DIR="$ROOT_DIR/BIGRec/results/$DATASET/$SAFE_MODEL_NAME/${BIGREC_SEED}_${BIGREC_SAMPLE}"
    RANKING_PATH="$RESULT_DIR/train_rank.txt"
    CONFIDENCE_PATH="$RESULT_DIR/train_score.txt"
    
    # Check if files exist (optional but good for debugging)
    if [ ! -f "$RANKING_PATH" ]; then
        echo "WARNING: Ranking file not found at $RANKING_PATH"
    fi
    
    EXTRA_ARGS="--embedding_path $EMBEDDING_PATH --ranking_path $RANKING_PATH --confidence_path $CONFIDENCE_PATH"
fi

# Define paths
DLLM2REC_DIR="DLLM2Rec"

cd "$DLLM2REC_DIR"

# Run training
SECONDS=0
python main.py \
    --data "$DATASET" \
    --model_name "$MODEL_NAME" \
    --alpha 0.5 \
    --ed_weight "$ED_WEIGHT" \
    --lam "$LAM" \
    --cuda "$GPU_ID" \
    --teacher_model "$BIGREC_BASE_MODEL" \
    --seed "$BIGREC_SEED" \
    --teacher_sample "$BIGREC_SAMPLE" \
    $EXTRA_ARGS

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
