#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-"game_bigrec"}

# Paths
ROOT_DIR=$(pwd)
BIGREC_DIR="$ROOT_DIR/BIGRec"
DATA_DIR="$BIGREC_DIR/data/$DATASET"
TRAIN_FILE="$DATA_DIR/train.json"
OUTPUT_FILE="$DATA_DIR/pop_count.json"

echo "Creating popularity file for dataset: $DATASET"
echo "Train file: $TRAIN_FILE"
echo "Output file: $OUTPUT_FILE"

# Check if train file exists
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Train file not found at $TRAIN_FILE"
    exit 1
fi

# Run python script
python "$BIGREC_DIR/data/create_pop_file.py" \
    --train_file "$TRAIN_FILE" \
    --output_file "$OUTPUT_FILE"

echo "Success! Popularity file created at $OUTPUT_FILE"
