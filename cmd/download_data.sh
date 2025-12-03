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
    # We might want to force check if it's the correct one, but for now rely on user to clean up if needed
    exit 0
fi

echo "Downloading MovieLens 10M dataset..."

# Download ML-10M
curl -O https://files.grouplens.org/datasets/movielens/ml-10m.zip

# Unzip
unzip -o ml-10m.zip

# Move ratings.dat and movies.dat
# ML-10M extracts to ml-10M100K/
echo "Moving ratings.dat and movies.dat to $DATA_DIR..."
mv ml-10M100K/ratings.dat "$DATA_DIR/"
mv ml-10M100K/movies.dat "$DATA_DIR/"

# Cleanup
rm -rf ml-10M100K ml-10m.zip

echo "Data download completed."
