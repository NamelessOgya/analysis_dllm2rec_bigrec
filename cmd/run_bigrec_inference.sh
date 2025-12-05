#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-movie}
GPU_ID=${2:-0}
BASE_MODEL=${3:-"Qwen/Qwen2-0.5B"}
SEED=${4:-0}
SAMPLE=${5:-1024}
SKIP_INFERENCE=${6:-false}
TARGET_SPLIT=${7:-"valid_test"}
BATCH_SIZE=${8:-16}
DEBUG_LIMIT=${9:--1}

echo "Running BIGRec inference for dataset: $DATASET"

# Sanitize model name for directory usage (replace / with _)
SAFE_MODEL_NAME=$(echo "$BASE_MODEL" | tr '/' '_')

# Define paths
BIGREC_DIR="BIGRec"
# Output directory structure aligned with training: ./model/<dataset>/<safe_model_name>/<seed>_<sample>
RESULT_DIR="BIGRec/results/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}"
# Define paths
# Use root-relative paths
DATA_DIR="BIGRec/data/$DATASET"

# Determine which files to process
FILES_TO_PROCESS=()
if [ "$TARGET_SPLIT" == "valid_test" ]; then
    FILES_TO_PROCESS+=("valid.json" "test.json")
elif [ "$TARGET_SPLIT" == "all" ]; then
    FILES_TO_PROCESS+=("valid.json" "test.json" "train.json")
else
    # Fallback: treat it as a specific filename if it's not a keyword
    FILES_TO_PROCESS+=("$TARGET_SPLIT")
fi

# Construct LoRA weights path
LORA_WEIGHTS="BIGRec/model/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}"

# Ensure result directory exists
mkdir -p "$RESULT_DIR"

# Check if LoRA weights exist
if [ ! -d "$LORA_WEIGHTS" ]; then
    echo "Error: LoRA weights not found at $LORA_WEIGHTS"
    echo "Please check your arguments (dataset, model, seed, sample) or run training first."
    exit 1
fi

# Check if item embedding file exists
EMBEDDING_DIR="BIGRec/data/$DATASET/model_embeddings"
EMBEDDING_FILE="$EMBEDDING_DIR/${SAFE_MODEL_NAME}.pt"

if [ ! -f "$EMBEDDING_FILE" ]; then
    echo "Item embedding file not found at $EMBEDDING_FILE"
    echo "Generating item embeddings..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python BIGRec/data/generate_embeddings.py \
        --dataset "$DATASET" \
        --base_model "$BASE_MODEL" \
        --output_path "$EMBEDDING_FILE"
    
    if [ ! -f "$EMBEDDING_FILE" ]; then
        echo "Error: Failed to generate item embeddings."
        exit 1
    fi
    echo "Item embeddings generated successfully."
fi

echo "Using LoRA weights from: $LORA_WEIGHTS"
echo "Outputting results to: $RESULT_DIR"

for TEST_DATA in "${FILES_TO_PROCESS[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Processing split: $TEST_DATA"
    echo "----------------------------------------------------------------"
    
    TEST_DATA_PATH="$DATA_DIR/$TEST_DATA"
    RESULT_JSON_PATH="$RESULT_DIR/$TEST_DATA"

    # Check if test data exists
    if [ ! -f "$TEST_DATA_PATH" ]; then
        echo "Error: Test data not found at $TEST_DATA_PATH"
        echo "Skipping..."
        continue
    fi

    # Run inference
    if [ "$SKIP_INFERENCE" = "true" ]; then
        echo "Skipping inference step as requested."
    else
        SECONDS=0
        CUDA_VISIBLE_DEVICES=$GPU_ID python BIGRec/inference.py \
            --base_model "$BASE_MODEL" \
            --lora_weights "$LORA_WEIGHTS" \
            --test_data_path "$TEST_DATA_PATH" \
            --result_json_data "$RESULT_JSON_PATH" \
            --batch_size "$BATCH_SIZE" \
            --limit "$DEBUG_LIMIT"
        duration=$SECONDS
        duration_min=$(($duration / 60))
        echo "Data generation for distillation time: $duration_min minutes"
        
        # Save/Update execution time to JSON
        python -c "import json; import os; 
path = os.path.join('$RESULT_DIR', 'execution_time.json');
data = {};
if os.path.exists(path):
    try:
        with open(path, 'r') as f: data = json.load(f)
    except: pass;
key = 'data_generation_time_minutes_' + '$TEST_DATA';
data[key] = $duration_min;
with open(path, 'w') as f: json.dump(data, f, indent=4)"
    fi

    echo "Inference completed (or skipped). Running evaluation..."

    # Run evaluation
    SECONDS=0
    CUDA_VISIBLE_DEVICES=$GPU_ID python "BIGRec/data/$DATASET/evaluate.py" \
        --input_file "$RESULT_JSON_PATH" \
        --base_model "$BASE_MODEL" \
        --embedding_path "$EMBEDDING_FILE" \
        --save_results \
        --batch_size "$BATCH_SIZE"
    duration=$SECONDS
    duration_min=$(($duration / 60))
    echo "Teacher model accuracy evaluation time: $duration_min minutes"

    # Save/Update execution time to JSON
    python -c "import json; import os; 
path = os.path.join('$RESULT_DIR', 'execution_time.json');
data = {};
if os.path.exists(path):
    try:
        with open(path, 'r') as f: data = json.load(f)
    except: pass;
key = 'evaluation_time_minutes_' + '$TEST_DATA';
data[key] = $duration_min;
with open(path, 'w') as f: json.dump(data, f, indent=4)"

    # evaluate.py writes output to ./<dataset>.json (e.g., movie.json) in the current directory.
    # Since we are in root, it will be ./movie.json
    # However, with --input_file, evaluate.py might behave differently regarding output?
    # Checking evaluate.py code:
    # result_dict[p] = ...
    # json.dump(result_dict, f, indent=4) where f = open('./game.json', 'w') (hardcoded output name based on dataset?)
    # The output filename in evaluate.py seems hardcoded to './game.json' (or whatever the script has).
    # We should probably move it to a unique name to avoid overwriting.
    
    if [ -f "./${DATASET}.json" ]; then
        mv "./${DATASET}.json" "$RESULT_DIR/${TEST_DATA}_metrics.json"
        echo "Evaluation metrics saved to $RESULT_DIR/${TEST_DATA}_metrics.json"
    else
        echo "Warning: Evaluation output file ./${DATASET}.json not found."
    fi
done

echo "BIGRec inference and evaluation completed for all requested splits."
