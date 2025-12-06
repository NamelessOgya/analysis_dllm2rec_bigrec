#!/bin/bash

# Exit on error
set -e

DATASET="game_bigrec"
SRC_DATASET="game" # Where raw data lives

echo "Preprocessing data for dataset: $DATASET"

# Define paths
BIGREC_DIR="BIGRec"
DATA_DIR="$BIGREC_DIR/data/$DATASET"
SCRIPT_PATH="$DATA_DIR/process.py"

# Check if python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    exit 1
fi

# Ensure output dir exists (it should if process.py is there)
mkdir -p "$DATA_DIR"

echo "Running data processing script..."
cd "$DATA_DIR"
# The script expects raw data in ../game/
python process.py

echo "Data preprocessing completed for $DATASET"
