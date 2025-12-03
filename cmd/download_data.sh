#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-movie}

echo "Checking data for dataset: $DATASET"

# Define paths
BIGREC_DIR="BIGRec"
DATA_DIR="$BIGREC_DIR/data/$DATASET"
RATINGS_FILE="$DATA_DIR/ratings.dat"

# Check if ratings.dat exists
if [ -f "$RATINGS_FILE" ]; then
    echo "ratings.dat already exists in $DATA_DIR"
    exit 0
fi

echo "ratings.dat not found. Downloading MovieLens 1M dataset..."

# Download ML-1M
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip

# Unzip
unzip -o ml-1m.zip

# Move ratings.dat
echo "Moving ratings.dat to $DATA_DIR..."
mv ml-1m/ratings.dat "$DATA_DIR/"

# Cleanup
rm -rf ml-1m ml-1m.zip

echo "Data download completed."
