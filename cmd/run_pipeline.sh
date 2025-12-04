#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-game}
GPU_ID=${2:-0}
BASE_MODEL=${3:-"Qwen/Qwen2-0.5B"}
SEED=${4:-0}
SAMPLE=${5:-1024}

echo "========================================================"
echo "Starting End-to-End Pipeline"
echo "Dataset: $DATASET"
echo "GPU ID: $GPU_ID"
echo "Base Model: $BASE_MODEL"
echo "========================================================"

# 1. Preprocessing
echo ">>> Step 1: Preprocessing"
if [ "$DATASET" == "game" ]; then
    ./cmd/run_preprocess_game.sh
elif [ "$DATASET" == "movie" ]; then
    ./cmd/run_preprocess_movie.sh
else
    ./cmd/run_preprocess_data.sh "$DATASET"
fi

# 2. BIGRec Training
echo ">>> Step 2: BIGRec Training"
# Usage: ./cmd/run_bigrec_train.sh <dataset> <gpu_id> <seed> <sample> <batch_size> <micro_batch_size> <base_model> <num_epochs>
# Using default batch sizes for now, can be parameterized if needed
./cmd/run_bigrec_train.sh "$DATASET" "$GPU_ID" "$SEED" "$SAMPLE" 128 4 "$BASE_MODEL" 50

# 3. BIGRec Inference (Generating Distillation Data)
echo ">>> Step 3: BIGRec Inference (on Training Data)"
# We need to run inference on 'train.json' to generate rank/confidence for DLLM2Rec training.
# Usage: ./cmd/run_bigrec_inference.sh <dataset> <gpu_id> <base_model> <seed> <sample> <skip_inference> <test_data> <batch_size>
./cmd/run_bigrec_inference.sh "$DATASET" "$GPU_ID" "$BASE_MODEL" "$SEED" "$SAMPLE" "false" "train.json" 32

# 4. Data Transfer
echo ">>> Step 4: Data Transfer"
# Usage: ./cmd/transfer_data.sh <dataset> <gpu_id> <base_model> <seed> <sample>
./cmd/transfer_data.sh "$DATASET" "$GPU_ID" "$BASE_MODEL" "$SEED" "$SAMPLE"

# 5. DLLM2Rec Training
echo ">>> Step 5: DLLM2Rec Training"
# Usage: ./cmd/run_dllm2rec_train.sh <dataset> <model_name> <gpu_id>
./cmd/run_dllm2rec_train.sh "$DATASET" "SASRec" "$GPU_ID"

echo "========================================================"
echo "Pipeline Completed Successfully!"
echo "========================================================"
