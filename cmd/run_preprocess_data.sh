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
SCRIPT_PATH="$DATA_DIR/process.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    exit 1
fi

# Ensure data exists
./cmd/download_data.sh "$DATASET"

# echo "Converting notebook to python script..."
# jupyter nbconvert --to python "$NOTEBOOK_PATH"

echo "Running data processing script..."
cd "$DATA_DIR"
python process.py

echo "Data preprocessing completed for $DATASET"
