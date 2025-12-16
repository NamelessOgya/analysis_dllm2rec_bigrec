#!/bin/bash

# Exit on error
set -e

# Arguments
# Arguments
DATASET=${1:-game}
GPU_ID=${2:-0}
EPOCH=${3:-200}

# Seed defaults to 2024
if [ -z "$4" ]; then
    SEED=2024
else
    SEED="$4"
fi

# Alpha (DROS weight) defaults to 0
if [ -z "$5" ]; then
    ALPHA=0
else
    ALPHA="$5"
fi

echo "Running SASRec Baseline on $DATASET with GPU=$GPU_ID, EPOCH=$EPOCH, SEED=$SEED, ALPHA=$ALPHA..."

# Define paths
# DLLM2REC_DIR="DLLM2Rec" # This variable is no longer used, direct path is used below

cd DLLM2Rec

# Run training
# Explicitly set ed_weight=0 and lam=0 to ensure we use "no_distillation" directory structure
# Add --export_train_scores to always export train.pt/train_uids.pt for potential BIGRec usage
SECONDS=0
# Handle output directory override
# Mandate output directory
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: OUTPUT_DIR environment variable is required."
    exit 1
fi
# Fix paths if they are relative to repo root (e.g. start with experiments/)
# because we will cd into DLLM2Rec
if [[ "$OUTPUT_DIR" == experiments/* ]]; then
    OUTPUT_DIR="../$OUTPUT_DIR"
fi

echo "Using output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
EXTRA_ARGS="--output_dir $OUTPUT_DIR"

python3 main.py \
    --data $DATASET \
    --model_name SASRec \
    --epoch $EPOCH \
    --alpha $ALPHA \
    --cuda $GPU_ID \
    --seed $SEED \
    --ed_weight 0 \
    --lam 0 \
    --export_train_scores \
    $EXTRA_ARGS

duration=$SECONDS
echo "SASRec baseline training completed in $(($duration / 60)) minutes and $(($duration % 60)) seconds"
