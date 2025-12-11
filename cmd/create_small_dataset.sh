#!/bin/bash

# Exit on error
set -e

# Dataset argument
DATASET=${1:-game}

echo "============================================"
echo "Creating small dataset for: $DATASET"
echo "============================================"

BIGREC_DIR="BIGRec"
DATA_DIR="$BIGREC_DIR/data/$DATASET"
PREPROCESS_SCRIPT="./cmd/run_preprocess_data.sh"

# Check if preprocess script exists
if [ ! -f "$PREPROCESS_SCRIPT" ]; then
    echo "Error: Preprocessing script not found at $PREPROCESS_SCRIPT"
    exit 1
fi

# Run preprocessing
# This script has been modified to automatically generate *_5000.json files
echo "Running preprocessing script..."
bash "$PREPROCESS_SCRIPT" "$DATASET"

# Verify outputs and create small dataset directory
echo "Verifying and restructuring small dataset..."
FILES=("train_5000.json" "valid_5000.json" "test_5000.json")
MISSING=0

for FILE in "${FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$FILE" ]; then
        echo "Error: $FILE not found in $DATA_DIR"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "Failed to create all small dataset files."
    exit 1
fi

# Create small dataset directory
SMALL_DATA_DIR="${DATA_DIR}_small"
echo "Creating small dataset directory: $SMALL_DATA_DIR"
mkdir -p "$SMALL_DATA_DIR"

# Copy and rename files
cp "$DATA_DIR/train_5000.json" "$SMALL_DATA_DIR/train.json"
cp "$DATA_DIR/valid_5000.json" "$SMALL_DATA_DIR/valid.json"
cp "$DATA_DIR/test_5000.json" "$SMALL_DATA_DIR/test.json"

# Copy auxiliary files if they exist
AUX_FILES=("id2name.txt" "movies.dat" "ratings.dat")
for AUX in "${AUX_FILES[@]}"; do
    if [ -f "$DATA_DIR/$AUX" ]; then
        echo "Copying $AUX..."
        cp "$DATA_DIR/$AUX" "$SMALL_DATA_DIR/"
    fi
done

echo "============================================"
echo "Small dataset creation completed successfully."
echo "Location: $SMALL_DATA_DIR"
echo "Files created:"
ls -1 "$SMALL_DATA_DIR"
echo "============================================"
