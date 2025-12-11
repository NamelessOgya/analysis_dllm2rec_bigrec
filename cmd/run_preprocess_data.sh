#!/bin/bash

# Exit on error
set -e

# Dataset argument (default to movie)
DATASET=${1:-movie}

echo "Preprocessing data for dataset: $DATASET"

# Define paths
BIGREC_DIR="BIGRec"
DATA_DIR="$BIGREC_DIR/data/$DATASET"
NOTEBOOK_PATH="$DATA_DIR/process.ipynb"
SCRIPT_PATH="$DATA_DIR/process.py"

# Check if python script exists
# Check if python script exists
if [ -f "$DATA_DIR/process.py" ]; then
    SCRIPT_PATH="$DATA_DIR/process.py"
elif [ -f "$DATA_DIR/convert_dllm2rec_to_bigrec.py" ]; then
    SCRIPT_PATH="$DATA_DIR/convert_dllm2rec_to_bigrec.py"
else
    echo "Error: No suitable python script found in $DATA_DIR"
    exit 1
fi

# Ensure data exists
./cmd/download_data.sh "$DATASET"

# echo "Converting notebook to python script..."
# jupyter nbconvert --to python "$NOTEBOOK_PATH"

echo "Running data processing script..."
cd "$DATA_DIR"
# python process.py
# python convert_dllm2rec_to_bigrec.py
python3 "$(basename "$SCRIPT_PATH")"

echo "Data preprocessing completed for $DATASET"
