#!/bin/bash

# Exit on error
set -e

# Default Arguments
DATASET="movie"
GPU_ID="0"
BASE_MODEL="Qwen/Qwen2-0.5B"
SEED="0"
SAMPLE="1024"
SKIP_INFERENCE="false"
BATCH_SIZE=16
LIMIT=-1
PROMPT_FILE=""
USE_EMBEDDING_MODEL="false"
CHECKPOINT_EPOCH="best"
TEST_DATA="test_5000.json" # Default for test_data

# Correction Modes
CORRECTION_MODE="none" # none, popularity, ci
CORRECTION_RESOURCE=""  # path to pop file (optional override) or CI score directory
MANUAL_GAMMA=""         # Optional manual gamma override

# Helper function for usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --dataset <name>         Dataset name (default: movie)"
    echo "  --gpu <ids>              GPU IDs (default: 0)"
    echo "  --model <name>           Base model name (default: Qwen/Qwen2-0.5B)"
    echo "  --seed <int>             Random seed (default: 0)"
    echo "  --sample <int>           Sample size (default: 1024)"
    echo "  --checkpoint <epoch>     Checkpoint epoch or 'best' (default: best)"
    echo "  --test_data <file/mode>  Test file, 'all', or 'valid_test' (default: test_5000.json)"
    echo "  --skip_inference         Skip inference, only evaluate (flag)"
    echo "  --limit <int>            Limit number of items (-1 for all)"
    echo "  --use_embedding_model    Use dedicated embedding model (flag)"
    echo "  --correction <type>      Correction type: 'none', 'popularity', 'ci' (default: none)"
    echo "  --resource <path>        Path to correction resource (CI dir or Pop file)"
    echo "  --gamma <float>          Manual gamma override (default: auto/grid-search)"
    echo "  --batch_size <int>       Batch size for evaluation (default: 16)"
    echo "  -h, --help               Show this help message and exit"
    exit 1
}

# Parse Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --gpu) GPU_ID="$2"; shift ;;
        --model) BASE_MODEL="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --sample) SAMPLE="$2"; shift ;;
        --checkpoint) CHECKPOINT_EPOCH="$2"; shift ;;
        --test_data) TEST_DATA="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;; # Included for compatibility
        --limit) LIMIT="$2"; shift ;;
        --prompt_file) PROMPT_FILE="$2"; shift ;;
        --skip_inference) SKIP_INFERENCE="true" ;;
        --use_embedding_model) USE_EMBEDDING_MODEL="true" ;;
        --correction) CORRECTION_MODE="$2"; shift ;;
        --resource) CORRECTION_RESOURCE="$2"; shift ;;
        --gamma) MANUAL_GAMMA="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

echo "========================================================"
echo "Running BIGRec Inference (vLLM)"
echo "Dataset: $DATASET"
echo "Model: $BASE_MODEL"
echo "Correction Mode: $CORRECTION_MODE"
echo "========================================================"

# Sanitize model name for directory usage (replace / with _)
SAFE_MODEL_NAME=$(echo "$BASE_MODEL" | tr '/' '_')

# Define paths
BIGREC_DIR="BIGRec"
RESULT_DIR="BIGRec/results/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}"
DATA_DIR="BIGRec/data/$DATASET"

# Construct LoRA weights path
BASE_LORA_PATH="BIGRec/model/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}"
LORA_WEIGHTS="$BASE_LORA_PATH"
EPOCH_SUFFIX="_epoch_best" # Default suffix

# Check for specific checkpoint
if [ -d "$BASE_LORA_PATH" ]; then
    if [ "$CHECKPOINT_EPOCH" == "best" ]; then
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
        TARGET_EPOCH="$CHECKPOINT_EPOCH"
        EPOCH_SUFFIX="_epoch${TARGET_EPOCH}"
        SPECIFIC_BEST="${BASE_LORA_PATH}/best_model_epoch_${TARGET_EPOCH}"
        if [ -d "$SPECIFIC_BEST" ]; then
             echo "Found specified epoch in best model: $SPECIFIC_BEST"
             LORA_WEIGHTS="$SPECIFIC_BEST"
        else
             FOUND_CHECKPOINT=""
             echo "Searching for epoch $TARGET_EPOCH in checkpoints..."
             
             for d in "$BASE_LORA_PATH"/checkpoint-*; do
                 if [ -d "$d" ]; then
                     STATE_FILE="$d/trainer_state.json"
                     if [ -f "$STATE_FILE" ]; then
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
    EXTRA_ARGS="--dataset $DATASET"
else
    # Check if test data exists (only for specific files)
    if [ ! -f "$TEST_DATA_PATH" ]; then
        echo "Error: Test data not found at $TEST_DATA_PATH"
        exit 1
    fi
    EXTRA_ARGS=""
    FILENAME=$(basename "$TEST_DATA")
    EXTENSION="${FILENAME##*.}"
    BASENAME="${FILENAME%.*}"
    RESULT_JSON_PATH="$RESULT_DIR/${BASENAME}${EPOCH_SUFFIX}.${EXTENSION}"
fi

# Ensure result directory exists
mkdir -p "$RESULT_DIR"

# Check LoRA
if [ ! -d "$LORA_WEIGHTS" ]; then
    echo "Error: LoRA weights not found at $LORA_WEIGHTS"
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
# Run inference with vLLM
if [ "$SKIP_INFERENCE" = "true" ]; then
    echo "Skipping inference step."
else
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_ID"
    NUM_GPUS=${#GPU_ARRAY[@]}
    echo "Using $NUM_GPUS GPUs: $GPU_ID"
    echo "Processing limit: $LIMIT"
    
    # Start timing for Inference
    START_TIME_INF=$(python3 -c 'import time; print(time.time())')

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

    # End timing for Inference
    END_TIME_INF=$(python3 -c 'import time; print(time.time())')
    ELAPSED_INF=$(python3 -c "print($END_TIME_INF - $START_TIME_INF)")
    ELAPSED_INF_MIN=$(python3 -c "print($ELAPSED_INF / 60)")
    echo "Data generation time: $ELAPSED_INF seconds ($ELAPSED_INF_MIN minutes)"

    # Save/Update execution time to JSON
    python -c "import json; import os; 
path = os.path.join('$RESULT_DIR', 'execution_time.json');
data = {};
if os.path.exists(path):
    try:
        with open(path, 'r') as f: data = json.load(f)
    except: pass;
key = 'data_generation_time_minutes_' + '$TEST_DATA';
data[key] = $ELAPSED_INF_MIN;
data[key.replace('_minutes_', '_seconds_')] = $ELAPSED_INF;
with open(path, 'w') as f: json.dump(data, f, indent=4)"
fi

echo "Inference completed. Running evaluation..."

# Correction Arguments Setup
EVAL_ARGS=""

# Logic for finding validation file for grid search
VALID_FILE=""
if [ -f "$RESULT_DIR/valid_epoch_best.json" ]; then VALID_FILE="$RESULT_DIR/valid_epoch_best.json"; fi
if [ -f "$RESULT_DIR/valid.json" ]; then VALID_FILE="$RESULT_DIR/valid.json"; fi
if [ -f "$RESULT_DIR/valid_test.json" ]; then VALID_FILE="$RESULT_DIR/valid_test.json"; fi 

if [ -n "$VALID_FILE" ]; then
    echo "Using Validation File for Grid Search: $VALID_FILE"
    EVAL_ARGS="$EVAL_ARGS --validation_file $VALID_FILE"
else
    echo "WARNING: No validation file found in $RESULT_DIR. Grid search may fail if enabled."
fi

# Manual Gamma
if [ -n "$MANUAL_GAMMA" ]; then
    EVAL_ARGS="$EVAL_ARGS --manual_gamma $MANUAL_GAMMA"
    echo "Using manual gamma: $MANUAL_GAMMA"
fi

if [ "$CORRECTION_MODE" == "popularity" ]; then
    POP_FILE="BIGRec/data/$DATASET/pop_count.json"
    if [ -n "$CORRECTION_RESOURCE" ]; then POP_FILE="$CORRECTION_RESOURCE"; fi
    
    if [ ! -f "$POP_FILE" ]; then
        echo "Generating popularity file..."
        if [ -f "$DATA_DIR/train.json" ]; then
            python BIGRec/data/create_pop_file.py --train_file "$DATA_DIR/train.json" --output_file "$POP_FILE"
            echo "Generated $POP_FILE"
        else
            echo "WARNING: Cannot generate popularity file. train.json not found in $DATA_DIR."
        fi
    fi
    
    if [ -f "$POP_FILE" ]; then
        EVAL_ARGS="$EVAL_ARGS --popularity_file $POP_FILE"
        echo "Correction: Popularity (File: $POP_FILE)"
    else
        echo "WARNING: Popularity file not found or generated. Skipping popularity correction."
    fi
    
elif [ "$CORRECTION_MODE" == "ci" ]; then
    CI_PATH="$CORRECTION_RESOURCE"
    if [ -z "$CI_PATH" ] || [ ! -d "$CI_PATH" ]; then
        echo "Error: CI Correction requires a valid directory via --resource. Provided: $CI_PATH"
        exit 1
    fi
    
    EVAL_ARGS="$EVAL_ARGS --ci_score_path $CI_PATH"
    echo "Correction: SASRec CI (Path: $CI_PATH)"
fi


# Run evaluation
# Run evaluation
# Start timing for Evaluation
START_TIME_EVAL=$(python3 -c 'import time; print(time.time())')

if [ "$TEST_DATA" = "all" ] || [ "$TEST_DATA" = "valid_test" ]; then
    # Directory mode
    CUDA_VISIBLE_DEVICES=$GPU_ID python "BIGRec/data/$DATASET/evaluate.py" \
        --input_dir "$RESULT_DIR" \
        --base_model "$EVAL_MODEL" \
        --embedding_path "$EMBEDDING_FILE" \
        --save_results \
        --batch_size "$BATCH_SIZE" \
        $EXTRA_EMBED_ARGS $EVAL_ARGS
else
    # Single file mode
    CUDA_VISIBLE_DEVICES=$GPU_ID python "BIGRec/data/$DATASET/evaluate.py" \
        --input_dir "$RESULT_DIR" \
        --base_model "$EVAL_MODEL" \
        --embedding_path "$EMBEDDING_FILE" \
        --save_results \
        --batch_size "$BATCH_SIZE" \
        --input_file "$RESULT_JSON_PATH" \
        $EXTRA_EMBED_ARGS $EVAL_ARGS
fi

# End timing for Evaluation
END_TIME_EVAL=$(python3 -c 'import time; print(time.time())')
ELAPSED_EVAL=$(python3 -c "print($END_TIME_EVAL - $START_TIME_EVAL)")
ELAPSED_EVAL_MIN=$(python3 -c "print($ELAPSED_EVAL / 60)")
echo "Evaluation time: $ELAPSED_EVAL seconds ($ELAPSED_EVAL_MIN minutes)"

# Save/Update execution time to JSON
python -c "import json; import os; 
path = os.path.join('$RESULT_DIR', 'execution_time.json');
data = {};
if os.path.exists(path):
    try:
        with open(path, 'r') as f: data = json.load(f)
    except: pass;
key = 'evaluation_time_minutes_' + '$TEST_DATA';
data[key] = $ELAPSED_EVAL_MIN;
data[key.replace('_minutes_', '_seconds_')] = $ELAPSED_EVAL;
with open(path, 'w') as f: json.dump(data, f, indent=4)"

# Saving metrics (backwards compat)
if [ -f "./${DATASET}.json" ]; then
    mv "./${DATASET}.json" "$RESULT_DIR/metrics.json"
fi
if [ -f "./best_gamma.txt" ]; then
    mv "./best_gamma.txt" "$RESULT_DIR/best_gamma.txt"
fi

echo "BIGRec inference (vLLM) and evaluation completed."
