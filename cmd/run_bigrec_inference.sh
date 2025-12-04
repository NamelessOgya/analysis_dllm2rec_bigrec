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

echo "Running BIGRec inference for dataset: $DATASET"

# Sanitize model name for directory usage (replace / with _)
SAFE_MODEL_NAME=$(echo "$BASE_MODEL" | tr '/' '_')

# Define paths
BIGREC_DIR="BIGRec"
# Output directory structure aligned with training: ./model/<dataset>/<safe_model_name>/<seed>_<sample>
# But for inference results, we might want to keep them separate or in the same place?
# The user asked to "separate output file destination by model like cmd/run_bigrec_train.sh".
# Training script uses: OUTPUT_DIR="./model/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}"
# Let's use a similar structure for results: ./results/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}
RESULT_DIR="./results/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}"
# TEST_DATA_PATH needs to be absolute or relative to BIGREC_DIR after cd
TEST_DATA_PATH="../data/$DATASET/test_5000.json"
RESULT_JSON_PATH="$RESULT_DIR/test.json"

# Construct LoRA weights path
# Path format: ./model/<dataset>/<safe_model_name>/<seed>_<sample>
LORA_WEIGHTS="./model/$DATASET/${SAFE_MODEL_NAME}/${SEED}_${SAMPLE}"

# Ensure result directory exists
mkdir -p "$BIGREC_DIR/$RESULT_DIR"

cd "$BIGREC_DIR"

# Check if LoRA weights exist
if [ ! -d "$LORA_WEIGHTS" ]; then
    echo "Error: LoRA weights not found at $LORA_WEIGHTS"
    echo "Please check your arguments (dataset, model, seed, sample) or run training first."
    exit 1
fi

echo "Using LoRA weights from: $LORA_WEIGHTS"
echo "Outputting results to: $RESULT_DIR"

# Run inference
if [ "$SKIP_INFERENCE" = "true" ]; then
    echo "Skipping inference step as requested."
else
    CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
        --base_model "$BASE_MODEL" \
        --lora_weights "$LORA_WEIGHTS" \
        --test_data_path "$TEST_DATA_PATH" \
        --result_json_data "$RESULT_JSON_PATH"
fi

echo "Inference completed (or skipped). Running evaluation..."

# Run evaluation
# evaluate.py takes --input_dir and processes all json files in it.
# We point it to our specific result directory.
CUDA_VISIBLE_DEVICES=$GPU_ID python "../data/$DATASET/evaluate.py" --input_dir "$RESULT_DIR" --base_model "$BASE_MODEL"

# evaluate.py writes output to ./<dataset>.json (e.g., movie.json) in the current directory.
# We move it to the result directory to keep things organized.
if [ -f "./${DATASET}.json" ]; then
    mv "./${DATASET}.json" "$RESULT_DIR/metrics.json"
    echo "Evaluation metrics saved to $RESULT_DIR/metrics.json"
else
    echo "Warning: Evaluation output file ./${DATASET}.json not found."
fi

echo "BIGRec inference and evaluation completed."
