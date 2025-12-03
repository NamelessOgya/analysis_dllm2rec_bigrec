#!/bin/bash

# Exit on error
set -e

SOURCE_DIR="BIGRec/data/movie"
TARGET_DIR="BIGRec/data/movie_small"
NUM_LINES=100000

echo "Creating small dataset in $TARGET_DIR..."

# Ensure source exists
if [ ! -f "$SOURCE_DIR/ratings.dat" ]; then
    echo "Error: Source ratings.dat not found in $SOURCE_DIR"
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_DIR"

# Copy movies.dat (it's small enough)
cp "$SOURCE_DIR/movies.dat" "$TARGET_DIR/"

# Create small ratings.dat
echo "Sampling top $NUM_LINES interactions..."
head -n $NUM_LINES "$SOURCE_DIR/ratings.dat" > "$TARGET_DIR/ratings.dat"

# Copy process.py if it exists, or we rely on the run script to use the one in source?
# The run script currently cd's to DATA_DIR. So we need process.py in TARGET_DIR.
if [ -f "$SOURCE_DIR/process.py" ]; then
    cp "$SOURCE_DIR/process.py" "$TARGET_DIR/"
fi

echo "Small dataset created successfully."
