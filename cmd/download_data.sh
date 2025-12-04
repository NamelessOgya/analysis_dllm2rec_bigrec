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

# Check if data already exists
if [ "$DATASET" == "movie" ]; then
    if [ -f "$RATINGS_FILE" ]; then
        echo "ratings.dat already exists in $DATA_DIR"
        exit 0
    fi

    echo "Downloading MovieLens 10M dataset..."
    curl -O https://files.grouplens.org/datasets/movielens/ml-10m.zip
    unzip -o ml-10m.zip
    
    echo "Moving ratings.dat and movies.dat to $DATA_DIR..."
    mv ml-10M100K/ratings.dat "$DATA_DIR/"
    mv ml-10M100K/movies.dat "$DATA_DIR/"
    
    rm -rf ml-10M100K ml-10m.zip

elif [ "$DATASET" == "game" ]; then
    REVIEWS_FILE="$DATA_DIR/Video_Games_5.json"
    META_FILE="$DATA_DIR/meta_Video_Games.json"

    if [ -f "$REVIEWS_FILE" ] && [ -f "$META_FILE" ]; then
        echo "Game data already exists in $DATA_DIR"
        exit 0
    fi

    echo "Downloading Amazon Video Games dataset..."
    # Using the links from the notebook or standard source
    # Note: These are large files, but smaller than movies
    curl -L -o "$DATA_DIR/Video_Games_5.json.gz" http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz
    curl -L -o "$DATA_DIR/meta_Video_Games.json.gz" http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Video_Games.json.gz

    echo "Unzipping game data..."
    gzip -d "$DATA_DIR/Video_Games_5.json.gz"
    gzip -d "$DATA_DIR/meta_Video_Games.json.gz"

else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

echo "Data download completed."
