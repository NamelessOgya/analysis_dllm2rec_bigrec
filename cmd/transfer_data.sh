#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-movie}
GPU_ID=${2:-0}
BASE_MODEL=${3:-"Qwen/Qwen2-0.5B"}
SEED=${4:-0}
SAMPLE=${5:-1024}

echo "Transferring data for dataset: $DATASET"
echo "GPU ID: $GPU_ID"
echo "Base Model: $BASE_MODEL"
echo "Seed: $SEED"
echo "Sample: $SAMPLE"

# Define paths
BIGREC_DIR="BIGRec"
DLLM2REC_DIR="DLLM2Rec"
TOCF_DIR="$DLLM2REC_DIR/tocf/$DATASET"

# Ensure destination directory exists
mkdir -p "$TOCF_DIR"

# Construct source directory path
# Replace / with _ in model name for directory safety
SAFE_MODEL_NAME=$(echo "$BASE_MODEL" | tr '/' '_')
SRC_DIR="$BIGREC_DIR/results/$DATASET/$SAFE_MODEL_NAME/${SEED}_${SAMPLE}"

echo "Source Directory: $SRC_DIR"

if [ -f "$SRC_DIR/all_embeddings.pt" ]; then
    cp "$SRC_DIR/all_embeddings.pt" "$TOCF_DIR/"
    echo "Copied all_embeddings.pt"
else
    echo "Warning: all_embeddings.pt not found in $SRC_DIR"
fi

if [ -f "$SRC_DIR/myrank_train.txt" ]; then
    cp "$SRC_DIR/myrank_train.txt" "$TOCF_DIR/"
    echo "Copied myrank_train.txt"
else
    echo "Warning: myrank_train.txt not found in $SRC_DIR"
fi

if [ -f "$SRC_DIR/confidence_train.txt" ]; then
    cp "$SRC_DIR/confidence_train.txt" "$TOCF_DIR/"
    echo "Copied confidence_train.txt"
else
    echo "Warning: confidence_train.txt not found in $SRC_DIR"
fi

echo "Data transfer completed."
