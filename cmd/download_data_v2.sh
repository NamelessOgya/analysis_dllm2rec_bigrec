#!/bin/bash

# Exit on error
set -e

DATA_DIR="BIGRec/data/game_v2"
mkdir -p "$DATA_DIR"

echo "Downloading Amazon V2 Game Data to $DATA_DIR..."

# URLs provided by user
REVIEW_URL="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz"
META_URL="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz"

echo "Downloading Reviews ($REVIEW_URL)..."
curl -L -o "$DATA_DIR/Video_Games_5.json.gz" "$REVIEW_URL"

echo "Downloading Metadata ($META_URL)..."
curl -L -o "$DATA_DIR/meta_Video_Games.json.gz" "$META_URL"

echo "Unzipping data..."
gzip -d "$DATA_DIR/Video_Games_5.json.gz"
gzip -d "$DATA_DIR/meta_Video_Games.json.gz"

echo "Download V2 completed."
