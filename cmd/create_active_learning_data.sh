#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-game_bigrec}
METHOD=${2:-loss} # random, pop_inverse, clustering, loss, entropy, error_rank
RATIO=${3:-0.5}
SEED=${4:-42}
BATCH_SIZE=${5:-1024}

# Optional Paths (can be overridden by environment variables or arguments if we extend this script, 
# but for now we follow the project structure)

echo "Running Active Learning Sampling for $DATASET with method=$METHOD, ratio=$RATIO, seed=$SEED..."

# Define paths
BASE_DIR="$(pwd)"
BIGREC_DIR="$BASE_DIR/BIGRec"
DLLM2Rec_DIR="$BASE_DIR/DLLM2Rec"

INPUT_JSON="$BIGREC_DIR/data/$DATASET/train.json"
INPUT_DF="$DLLM2Rec_DIR/data/$DATASET/train_data.df"

# DROS Outputs (Default location for SASRec / alpha=1.0 / seed=2024 / epoch=200)
# Adjust these defaults if your DROS training settings differ!
DROS_DIR="$DLLM2Rec_DIR/results/$DATASET/sasrec_no_distillation/2024/alpha_1.0"
DROS_SCORE="$DROS_DIR/train.pt"
DROS_UID="$DROS_DIR/train_uids.pt"
ITEM_EMB="$DLLM2Rec_DIR/tocf/$DATASET/all_embeddings.pt"

OUTPUT_JSON="$BIGREC_DIR/data/$DATASET/train_${METHOD}_${RATIO}.json"

# Check if data exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "Error: Input JSON not found at $INPUT_JSON"
    exit 1
fi

# Construct command
CMD="python3 $BIGREC_DIR/data/game_bigrec/sample_data.py \
    --input_json $INPUT_JSON \
    --input_df $INPUT_DF \
    --method $METHOD \
    --ratio $RATIO \
    --output_json $OUTPUT_JSON \
    --seed $SEED \
    --batch_size $BATCH_SIZE"

# Add DROS args if needed
if [[ "$METHOD" == "loss" || "$METHOD" == "entropy" || "$METHOD" == "error_rank" ]]; then
    if [ ! -f "$DROS_SCORE" ]; then
        echo "Error: DROS score file not found at $DROS_SCORE. Please run SASRec baseline first."
        exit 1
    fi
    CMD="$CMD --dros_score $DROS_SCORE --dros_uid $DROS_UID"
fi

# Add Clustering args if needed
if [[ "$METHOD" == "clustering" ]]; then
    if [ ! -f "$ITEM_EMB" ]; then
        echo "Error: Item embeddings not found at $ITEM_EMB."
        exit 1
    fi
    CMD="$CMD --item_emb $ITEM_EMB"
fi

echo "Executing: $CMD"
$CMD

echo "Done. Created $OUTPUT_JSON"
