#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-game_bigrec}
METHOD=${2:-random} # random, pop_inverse, clustering, loss, entropy, error_rank
SAMPLE_NUM=${3:-1000}
AL_RATIO=${4:-1.0}
SEED=${5:-42}
BATCH_SIZE=${6:-1024}
DROS_SOURCE=${7:-""}
MIN_RANK=${8:-10}
MAX_RANK=${9:-100}

echo "Dataset: $DATASET"
echo "Method: $METHOD"
echo "Sample Num: $SAMPLE_NUM"
echo "AL Ratio: $AL_RATIO"

# Optional Paths (can be overridden by environment variables or arguments if we extend this script, 
# but for now we follow the project structure)

echo "Running Active Learning Sampling for $DATASET with method=$METHOD, sample_num=$SAMPLE_NUM, al_ratio=$AL_RATIO, seed=$SEED..."

# Define paths
BASE_DIR="$(pwd)"
BIGREC_DIR="$BASE_DIR/BIGRec"
DLLM2Rec_DIR="$BASE_DIR/DLLM2Rec"

INPUT_JSON="$BIGREC_DIR/data/$DATASET/train.json"
INPUT_DF="$DLLM2Rec_DIR/data/$DATASET/train_data.df"

# DROS Outputs (Default location for SASRec / alpha=1.0 / seed=2024 / epoch=200)
# Adjust these defaults if your DROS training settings differ!
if [ -z "$DROS_SOURCE" ]; then
    DROS_DIR="$DLLM2Rec_DIR/results/$DATASET/sasrec_no_distillation/2024/alpha_1.0"
else
    DROS_DIR="$DROS_SOURCE"
fi

DROS_SCORE="$DROS_DIR/train.pt"
DROS_UID="$DROS_DIR/train_uids.pt"
ITEM_EMB="$DLLM2Rec_DIR/tocf/$DATASET/all_embeddings.pt"

# Construct Filename
MODEL_SUFFIX=""
if [[ "$METHOD" == "loss" || "$METHOD" == "entropy" || "$METHOD" == "error_rank" ]]; then
    # Extract alpha or directory name as model identifier
    MODEL_SUFFIX="_$(basename "$DROS_DIR")"
fi

RATIO_SUFFIX=""
if [ "$AL_RATIO" != "1.0" ]; then
    RATIO_SUFFIX="_ratio${AL_RATIO}"
fi

OUTPUT_JSON="$BIGREC_DIR/data/$DATASET/train_${METHOD}_${SAMPLE_NUM}${RATIO_SUFFIX}_seed${SEED}${MODEL_SUFFIX}.json"

# Check if data exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "Error: Input JSON not found at $INPUT_JSON"
    exit 1
fi

# Construct command
CMD="python3 $BIGREC_DIR/data/$DATASET/sample_data.py \
    --input_json $INPUT_JSON \
    --input_df $INPUT_DF \
    --method $METHOD \
    --sample_num $SAMPLE_NUM \
    --al_ratio $AL_RATIO \
    --output_json $OUTPUT_JSON \
    --seed $SEED \
    --batch_size $BATCH_SIZE"

# Add DROS args if needed
if [[ "$METHOD" == "loss" || "$METHOD" == "entropy" || "$METHOD" == "error_rank" || "$METHOD" == "proximal_rank" || "$METHOD" == "semantic_loss" || "$METHOD" == "confident_error" ]]; then
    if [ ! -f "$DROS_SCORE" ]; then
        echo "Error: DROS score file not found at $DROS_SCORE. Please run SASRec baseline first."
        exit 1
    fi
    CMD="$CMD --dros_score $DROS_SCORE --dros_uid $DROS_UID"
fi

# Add Embedding args if needed
if [[ "$METHOD" == "clustering" || "$METHOD" == "semantic_loss" ]]; then
    if [ ! -f "$ITEM_EMB" ]; then
        echo "Error: Item embeddings not found at $ITEM_EMB."
        exit 1
    fi
    CMD="$CMD --item_emb $ITEM_EMB"
fi

echo "Executing: $CMD"
$CMD

echo "Done. Created $OUTPUT_JSON"
