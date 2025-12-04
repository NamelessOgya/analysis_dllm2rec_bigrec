#!/bin/bash
set -e

# Test parameters
DATASET="movie"
GPU_ID=0
SEED=42
SAMPLE=10
BATCH_SIZE=2
MICRO_BATCH_SIZE=1
BASE_MODEL="Qwen/Qwen2-0.5B"
NUM_EPOCHS=1

echo "Starting verification run..."

# Run training script with minimal parameters
./cmd/run_bigrec_train.sh "$DATASET" "$GPU_ID" "$SEED" "$SAMPLE" "$BATCH_SIZE" "$MICRO_BATCH_SIZE" "$BASE_MODEL" "$NUM_EPOCHS"

# Expected output directory
SAFE_MODEL_NAME=$(echo "$BASE_MODEL" | tr '/' '_')
EXPECTED_DIR="$(pwd)/BIGRec/model/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}"

echo "Checking for output directory: $EXPECTED_DIR"

if [ -d "$EXPECTED_DIR" ]; then
    echo "Directory exists."
    if [ -f "$EXPECTED_DIR/adapter_model.bin" ] || [ -f "$EXPECTED_DIR/adapter_model.safetensors" ]; then
        echo "Success: Model file found."
        ls -l "$EXPECTED_DIR"
    else
        echo "Failure: Model file NOT found in $EXPECTED_DIR"
        ls -l "$EXPECTED_DIR"
        exit 1
    fi
else
    echo "Failure: Directory $EXPECTED_DIR does not exist."
    exit 1
fi
