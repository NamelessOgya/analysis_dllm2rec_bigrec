#!/bin/bash

# Exit on error
set -e

SOURCE_DIR="BIGRec/data/game"
TARGET_DIR="BIGRec/data/debug_single_game"
NUM_ITEMS=10 # Number of items to extract for debug dataset

echo "Creating debug dataset in $TARGET_DIR..."

# Create target directory
mkdir -p "$TARGET_DIR"

# Check if source train.json exists
if [ ! -f "$SOURCE_DIR/train.json" ]; then
    echo "Error: Source file $SOURCE_DIR/train.json does not exist."
    exit 1
fi

# Extract first N items from train.json to create a small dataset
# We use python for reliable JSON handling
python3 -c "
import json
import random

with open('$SOURCE_DIR/train.json', 'r') as f:
    data = json.load(f)

# Take first N items
small_data = data[:$NUM_ITEMS]

# Save to all split files in target
for filename in ['train.json', 'valid.json', 'test.json', 'test_5000.json', 'valid_5000.json']:
    with open(f'$TARGET_DIR/{filename}', 'w') as f:
        json.dump(small_data, f, indent=4)
"

echo "Created JSON data files with $NUM_ITEMS items."

# Copy configuration and helper files
echo "Copying auxiliary files..."
cp "$SOURCE_DIR/id2name.txt" "$TARGET_DIR/"
# evaluate.py is copied/patched below

# Copy model embeddings if they exist
if [ -d "$SOURCE_DIR/model_embeddings" ]; then
    cp -r "$SOURCE_DIR/model_embeddings" "$TARGET_DIR/"
    echo "Copied model_embeddings."
else
    echo "Warning: model_embeddings directory not found in source."
fi

# Copy dummy csv files just in case script checks them (though valid_5000.json is usually prioritized)
# Creating dummy CSVs if original CSVs exist is safer, but for now we focus on JSONs which are used by train.py/inference.py
# If CSVs are needed by evaluate.py, we might need them. Let's start with JSONs.

# Patch evaluate.py to output debug_single_game.json instead of game.json
# This is required because run_bigrec_inference*.sh expects <dataset>.json
echo "Patching evaluate.py..."
sed "s/game.json/debug_single_game.json/g" "$SOURCE_DIR/evaluate.py" > "$TARGET_DIR/evaluate.py"


echo "Debug dataset creation completed successfully."
ls -l "$TARGET_DIR"
