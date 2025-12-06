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

# Check if raw data exists, if not download it
RAW_DATA_DIR="$BIGREC_DIR/data/game_v2"
if [ ! -f "$RAW_DATA_DIR/Video_Games_5.json" ] || [ ! -f "$RAW_DATA_DIR/meta_Video_Games.json" ]; then
    echo "Raw data not found in $RAW_DATA_DIR. Running download script..."
    ./cmd/download_data_v2.sh
else
    echo "Raw data found in $RAW_DATA_DIR."
fi

# Ensure output dir exists (it should if process.py is there)
mkdir -p "$DATA_DIR"

echo "Running data processing script..."
cd "$DATA_DIR"
# The script expects raw data in ../game/
python process.py


echo "Data preprocessing completed for $DATASET"

# Copy data to DLLM2Rec directory for training
DLLM2REC_DATA_DIR="../../../DLLM2Rec/data/$DATASET"
echo "Copying data to DLLM2Rec directory: $DLLM2REC_DATA_DIR"
mkdir -p "$DLLM2REC_DATA_DIR"

cp train_data.df "$DLLM2REC_DATA_DIR/"
cp val_data.csv "$DLLM2REC_DATA_DIR/"
cp test_data.csv "$DLLM2REC_DATA_DIR/"
cp data_statis.df "$DLLM2REC_DATA_DIR/"
cp id2name.txt "$DLLM2REC_DATA_DIR/"

echo "Data transfer to DLLM2Rec completed."
