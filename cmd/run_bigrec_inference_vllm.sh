#!/bin/bash

# Exit on error
set -e

# Arguments
# Arguments
DATASET=${1:-movie}
GPU_ID=${2:-0}
BASE_MODEL=${3:-"Qwen/Qwen2-0.5B"}
SEED=${4:-0}
SAMPLE=${5:-1024}
SKIP_INFERENCE=${6:-false}
TEST_DATA=${7:-"test_5000.json"}
BATCH_SIZE=${8:-16} # Kept for compatibility, though vLLM manages it internally
LIMIT=${9:--1}      # New argument: Limit number of items to process (-1 for all)
PROMPT_FILE=${10:-""}
USE_EMBEDDING_MODEL=${11:-false}
USE_POPULARITY=${12:-false}
POPULARITY_GAMMA=${13:-0.0}
CHECKPOINT_EPOCH=${14:-"best"}

echo "Running BIGRec inference (vLLM) for dataset: $DATASET"

# Sanitize model name for directory usage (replace / with _)
SAFE_MODEL_NAME=$(echo "$BASE_MODEL" | tr '/' '_')

# Define paths
BIGREC_DIR="BIGRec"
RESULT_DIR="BIGRec/results/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}"
DATA_DIR="BIGRec/data/$DATASET"
# Result path calculation deferred until suffix is known

# Construct LoRA weights path
BASE_LORA_PATH="BIGRec/model/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}"
LORA_WEIGHTS="$BASE_LORA_PATH"
EPOCH_SUFFIX="_epoch_best" # Default suffix

# Check for specific checkpoint
if [ -d "$BASE_LORA_PATH" ]; then
    if [ "$CHECKPOINT_EPOCH" == "best" ]; then
        # Find directory named best_model_epoch_*
        # We use sort to pick the one with highest epoch if multiple exist (though training usually saves one 'best')
        BEST_MODEL=$(find "$BASE_LORA_PATH" -maxdepth 1 -type d -name "best_model_epoch_*" | sort -V | tail -n 1)
        
        if [ -n "$BEST_MODEL" ]; then
            echo "Found Best Model checkpoint: $BEST_MODEL"
            LORA_WEIGHTS="$BEST_MODEL"
            EPOCH_SUFFIX="_epoch_best"
        else
            echo "Error: No best model checkpoint (best_model_epoch_*) found in $BASE_LORA_PATH"
            exit 1
        fi
    else
        # User specified a specific epoch
        TARGET_EPOCH="$CHECKPOINT_EPOCH"
        EPOCH_SUFFIX="_epoch${TARGET_EPOCH}"
        
        # 1. Check if best_model_epoch_X exists
        SPECIFIC_BEST="${BASE_LORA_PATH}/best_model_epoch_${TARGET_EPOCH}"
        if [ -d "$SPECIFIC_BEST" ]; then
             echo "Found specified epoch in best model: $SPECIFIC_BEST"
             LORA_WEIGHTS="$SPECIFIC_BEST"
        else
             # 2. Search in checkpoint-* directories via trainer_state.json
             FOUND_CHECKPOINT=""
             echo "Searching for epoch $TARGET_EPOCH in checkpoints..."
             
             for d in "$BASE_LORA_PATH"/checkpoint-*; do
                 if [ -d "$d" ]; then
                     STATE_FILE="$d/trainer_state.json"
                     if [ -f "$STATE_FILE" ]; then
                         # python one-liner to extract epoch (as float)
                         CHECK_EPOCH=$(python -c "import json; 
try:
    with open('$STATE_FILE') as f: data = json.load(f); 
    print(data.get('epoch', -1))
except: print(-1)")
                         IS_MATCH=$(python -c "print(1 if abs(float('$CHECK_EPOCH') - float('$TARGET_EPOCH')) < 0.001 else 0)")
                         if [ "$IS_MATCH" -eq 1 ]; then
                             FOUND_CHECKPOINT="$d"
                             break
                         fi
                     fi
                 fi
             done
             
             if [ -n "$FOUND_CHECKPOINT" ]; then
                  echo "Found checkpoint for epoch $TARGET_EPOCH: $FOUND_CHECKPOINT"
                  LORA_WEIGHTS="$FOUND_CHECKPOINT"
             else
                  echo "Error: Could not find checkpoint for epoch $TARGET_EPOCH in $BASE_LORA_PATH"
                  exit 1
             fi
        fi
    fi
fi

# Define paths now that suffix is known
TEST_DATA_PATH="$DATA_DIR/$TEST_DATA"

# Special handling for "all" or "valid_test"
if [ "$TEST_DATA" = "all" ] || [ "$TEST_DATA" = "valid_test" ]; then
    TEST_DATA_PATH="$TEST_DATA"
    RESULT_JSON_PATH="$RESULT_DIR" # Output to directory
    
    # Pass dataset argument required by inference_vllm.py for these modes
    EXTRA_ARGS="--dataset $DATASET"
else
    # Check if test data exists (only for specific files)
    if [ ! -f "$TEST_DATA_PATH" ]; then
        echo "Error: Test data not found at $TEST_DATA_PATH"
        echo "Please run data preprocessing first (e.g., ./cmd/run_preprocess_data.sh $DATASET)."
        exit 1
    fi
    EXTRA_ARGS=""
    
    # Single file mode: Append suffix to filename
    # Extract extension and base
    FILENAME=$(basename "$TEST_DATA")
    EXTENSION="${FILENAME##*.}"
    BASENAME="${FILENAME%.*}"
    RESULT_JSON_PATH="$RESULT_DIR/${BASENAME}${EPOCH_SUFFIX}.${EXTENSION}"
fi

# Ensure result directory exists
mkdir -p "$RESULT_DIR"

# Check if LoRA weights exist
if [ ! -d "$LORA_WEIGHTS" ]; then
    echo "Error: LoRA weights not found at $LORA_WEIGHTS"
    echo "Please check your arguments (dataset, model, seed, sample) or run training first."
    exit 1
fi

echo "Using LoRA weights from: $LORA_WEIGHTS"
echo "Outputting results to: $RESULT_DIR"

# Configure Embedding Model
if [ "$USE_EMBEDDING_MODEL" = "true" ]; then
    EVAL_MODEL="intfloat/multilingual-e5-large"
    SAFE_EVAL_MODEL_NAME=$(echo "$EVAL_MODEL" | tr '/' '_')
    EXTRA_EMBED_ARGS="--use_embedding_model"
    echo "Using dedicated embedding model: $EVAL_MODEL"
else
    EVAL_MODEL="$BASE_MODEL"
    SAFE_EVAL_MODEL_NAME=$(echo "$EVAL_MODEL" | tr '/' '_')
    EXTRA_EMBED_ARGS=""
    echo "Using base model for embeddings: $EVAL_MODEL"
fi

# Check if item embedding file exists
EMBEDDING_DIR="BIGRec/data/$DATASET/model_embeddings"
EMBEDDING_FILE="$EMBEDDING_DIR/${SAFE_EVAL_MODEL_NAME}.pt"

if [ ! -f "$EMBEDDING_FILE" ]; then
    echo "Item embedding file not found at $EMBEDDING_FILE"
    echo "Generating item embeddings..."
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    python BIGRec/data/generate_embeddings.py \
        --dataset "$DATASET" \
        --base_model "$EVAL_MODEL" \
        --output_path "$EMBEDDING_FILE" \
        $EXTRA_EMBED_ARGS
    
    if [ ! -f "$EMBEDDING_FILE" ]; then
        echo "Error: Failed to generate item embeddings."
        exit 1
    fi
    echo "Item embeddings generated successfully."
fi

# Run inference with vLLM
if [ "$SKIP_INFERENCE" = "true" ]; then
    echo "Skipping inference step as requested."
else
    # Calculate number of GPUs
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_ID"
    NUM_GPUS=${#GPU_ARRAY[@]}
    echo "Using $NUM_GPUS GPUs: $GPU_ID"
    echo "Processing limit: $LIMIT"

    # Note: vLLM manages GPU memory aggressively. We set CUDA_VISIBLE_DEVICES.
    CUDA_VISIBLE_DEVICES=$GPU_ID python BIGRec/inference_vllm.py \
        --base_model "$BASE_MODEL" \
        --lora_weights "$LORA_WEIGHTS" \
        --test_data_path "$TEST_DATA_PATH" \
        --result_json_data "$RESULT_JSON_PATH" \
        --batch_size "$BATCH_SIZE" \
        --tensor_parallel_size "$NUM_GPUS" \
        --limit "$LIMIT" \
        --output_suffix "$EPOCH_SUFFIX" \
        $( [ -n "$PROMPT_FILE" ] && echo "--prompt_file $PROMPT_FILE" ) \
        $EXTRA_ARGS
fi

echo "Inference completed (or skipped). Running evaluation..."

# Popularity Arguments Setup
POP_ARGS=""
if [ "$USE_POPULARITY" = "true" ]; then
    POP_FILE="BIGRec/data/$DATASET/pop_count.json"
    
    # Check and generate if missing
    if [ ! -f "$POP_FILE" ]; then
        echo "Popularity file $POP_FILE not found. Generating..."
        if [ -f "$DATA_DIR/train.json" ]; then
            python BIGRec/data/create_pop_file.py --train_file "$DATA_DIR/train.json" --output_file "$POP_FILE"
            echo "Generated $POP_FILE"
        else
            echo "WARNING: Cannot generate popularity file. train.json not found in $DATA_DIR."
        fi
    fi
    
    if [ -f "$POP_FILE" ]; then
            POP_ARGS="--popularity_file $POP_FILE --popularity_gamma $POPULARITY_GAMMA"
            echo "Using Popularity Adjustment with gamma=$POPULARITY_GAMMA"
    fi
fi

# Run evaluation
if [ "$TEST_DATA" = "all" ] || [ "$TEST_DATA" = "valid_test" ]; then
    # Directory mode
    CUDA_VISIBLE_DEVICES=$GPU_ID python "BIGRec/data/$DATASET/evaluate.py" \
        --input_dir "$RESULT_DIR" \
        --base_model "$EVAL_MODEL" \
        --embedding_path "$EMBEDDING_FILE" \
        --save_results \
        --batch_size "$BATCH_SIZE" \
        $EXTRA_EMBED_ARGS $POP_ARGS
else
    # Single file mode
    CUDA_VISIBLE_DEVICES=$GPU_ID python "BIGRec/data/$DATASET/evaluate.py" \
        --input_dir "$RESULT_DIR" \
        --base_model "$EVAL_MODEL" \
        --embedding_path "$EMBEDDING_FILE" \
        --save_results \
        --batch_size "$BATCH_SIZE" \
        --input_file "$RESULT_JSON_PATH" \
        $EXTRA_EMBED_ARGS $POP_ARGS
fi

if [ -f "./${DATASET}.json" ]; then
    mv "./${DATASET}.json" "$RESULT_DIR/metrics.json"
    echo "Evaluation metrics saved to $RESULT_DIR/metrics.json"
else
    echo "Warning: Evaluation output file ./${DATASET}.json not found."
fi

echo "BIGRec inference (vLLM) and evaluation completed."
