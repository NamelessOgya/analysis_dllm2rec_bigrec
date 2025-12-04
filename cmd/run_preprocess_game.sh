#!/bin/bash

# Exit on error
set -e

DATASET="game"

echo "Preprocessing data for dataset: $DATASET"

# Define paths
BIGREC_DIR="BIGRec"
DATA_DIR="$BIGREC_DIR/data/$DATASET"
SCRIPT_PATH="$DATA_DIR/convert_dllm2rec_to_bigrec.py"

# Check if python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    exit 1
fi

# Ensure data exists (using the shared download script)
./cmd/download_data.sh "$DATASET"

echo "Running data processing script..."
cd "$DATA_DIR"
python convert_dllm2rec_to_bigrec.py

echo "Data preprocessing completed for $DATASET"
