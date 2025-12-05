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
TEST_DATA=${7:-"test_5000.json"}
BATCH_SIZE=${8:-16} # Kept for compatibility, though vLLM manages it internally

echo "Running BIGRec inference (vLLM) for dataset: $DATASET"

# Sanitize model name for directory usage (replace / with _)
SAFE_MODEL_NAME=$(echo "$BASE_MODEL" | tr '/' '_')

# Define paths
BIGREC_DIR="BIGRec"
RESULT_DIR="BIGRec/results/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}"
DATA_DIR="BIGRec/data/$DATASET"
TEST_DATA_PATH="$DATA_DIR/$TEST_DATA"
RESULT_JSON_PATH="$RESULT_DIR/$TEST_DATA"

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

# Check if test data exists
if [ ! -f "$TEST_DATA_PATH" ]; then
    echo "Error: Test data not found at $TEST_DATA_PATH"
    echo "Please run data preprocessing first (e.g., ./cmd/run_preprocess_data.sh $DATASET)."
    exit 1
fi

echo "Using LoRA weights from: $LORA_WEIGHTS"
echo "Outputting results to: $RESULT_DIR"

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

# Run inference with vLLM
if [ "$SKIP_INFERENCE" = "true" ]; then
    echo "Skipping inference step as requested."
else
    # Calculate number of GPUs
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_ID"
    NUM_GPUS=${#GPU_ARRAY[@]}
    echo "Using $NUM_GPUS GPUs: $GPU_ID"

    # Note: vLLM manages GPU memory aggressively. We set CUDA_VISIBLE_DEVICES.
    CUDA_VISIBLE_DEVICES=$GPU_ID python BIGRec/inference_vllm.py \
        --base_model "$BASE_MODEL" \
        --lora_weights "$LORA_WEIGHTS" \
        --test_data_path "$TEST_DATA_PATH" \
        --result_json_data "$RESULT_JSON_PATH" \
        --batch_size "$BATCH_SIZE" \
        --tensor_parallel_size "$NUM_GPUS"
fi

echo "Inference completed (or skipped). Running evaluation..."

# Run evaluation (unchanged)
CUDA_VISIBLE_DEVICES=$GPU_ID python "BIGRec/data/$DATASET/evaluate.py" \
    --input_dir "$RESULT_DIR" \
    --base_model "$BASE_MODEL" \
    --embedding_path "$EMBEDDING_FILE" \
    --save_results \
    --batch_size "$BATCH_SIZE"

if [ -f "./${DATASET}.json" ]; then
    mv "./${DATASET}.json" "$RESULT_DIR/metrics.json"
    echo "Evaluation metrics saved to $RESULT_DIR/metrics.json"
else
    echo "Warning: Evaluation output file ./${DATASET}.json not found."
fi

echo "BIGRec inference (vLLM) and evaluation completed."
