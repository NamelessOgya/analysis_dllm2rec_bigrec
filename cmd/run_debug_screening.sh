#!/bin/bash

# Exit on error
set -e

# Arguments
GPU_ID=${1:-0}

echo "Starting Screening Check..."

# 1. Create Debug Dataset
echo "Step 1: Creating Debug Dataset..."
./cmd/create_debug_dataset.sh

# 2. Run Training on Debug Dataset
echo "Step 2: Training on Debug Dataset..."
# Use a small number of epochs (e.g. 10) just to check if it runs and overfits
# Arguments: dataset gpu seed sample batch micro_batch model epochs
./cmd/run_bigrec_train.sh debug_single_game $GPU_ID 42 -1 2 1 "Qwen/Qwen2-0.5B" 10

# 3. Run Inference on Debug Dataset
echo "Step 3: Inference on Debug Dataset..."
# Arguments: dataset gpu model seed sample skip_inference test_data batch limit prompt_file use_embedding
# We test on train.json (which is the same as test/valid in this debug set) to check overfitting
./cmd/run_bigrec_inference_vllm.sh debug_single_game $GPU_ID "Qwen/Qwen2-0.5B" 42 1024 false "train.json" 2 -1 "" false

echo "Screening Check Completed."
echo "Please check the results in BIGRec/results/debug_single_game/"
