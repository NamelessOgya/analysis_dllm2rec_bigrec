#!/bin/bash

# Exit on error
set -e

DATASET="movie_bigrec"

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

# Ensure output dir exists
mkdir -p "$DATA_DIR"

# Ensure raw data exists using the shared download script (from 'movie' dataset)
echo "Ensuring raw MovieLens data exists..."
./cmd/download_data.sh movie

echo "Running data processing script..."
cd "$DATA_DIR"
# The script expects raw data in ../movie/
python3 process.py


echo "Data preprocessing completed for $DATASET"

# Generate DLLM2Rec data using the robust conversion script
echo "Generating aligned data for DLLM2Rec..."
# We are in BIGRec/data/movie_bigrec/, need to go back to root
cd "../../.."
python3 DLLM2Rec/convert_bigrec_data.py --dataset $DATASET

echo "Data preprocessing and conversion completed."
