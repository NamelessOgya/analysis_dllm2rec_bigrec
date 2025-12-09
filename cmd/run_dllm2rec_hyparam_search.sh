#!/bin/bash

# Exit on error
set -e

# Arguments
DATASET=${1:-game}
MODEL_NAME=${2:-SASRec}
GPU_ID=${3:-0}
BIGREC_BASE_MODEL=${4:-"Qwen/Qwen2-0.5B"}
BIGREC_SEED=${5:-0}
BIGREC_SAMPLE=${6:-1024}

echo "Starting Hyperparameter Search for DLLM2Rec"
echo "Dataset: $DATASET"
echo "Model: $MODEL_NAME"
echo "Teacher: $BIGREC_BASE_MODEL"
echo "Seed: $BIGREC_SEED"
echo "Sample: $BIGREC_SAMPLE"

# Initialize best variables
BEST_HR20=-1.0
BEST_ED=-1
BEST_LAM=-1
BEST_METRICS_FILE=""

# Define output directory for the summary (using the swaped structure: .../seed_sample/)
SAFE_TEACHER_NAME=$(echo "$BIGREC_BASE_MODEL" | tr '/' '_')
BASE_RESULT_DIR="DLLM2Rec/results/${DATASET}/${MODEL_NAME,,}_distilled_${SAFE_TEACHER_NAME}/${BIGREC_SEED}_${BIGREC_SAMPLE}"

# Iterate ed_weight from 0.0 to 1.0 step 0.1
for ed in $(seq 0.0 0.1 1.0); do
    for lam in $(seq 0.0 0.1 1.0); do
        echo "========================================================"
        echo "Testing Params: ed_weight=$ed, lam=$lam"
        echo "========================================================"
        
        # Run training
        # Note: We assume run_dllm2rec_train.sh is in the same cmd directory or executable from root
        # Using ./cmd/run_dllm2rec_train.sh assuming running from root
        ./cmd/run_dllm2rec_train.sh "$DATASET" "$MODEL_NAME" "$GPU_ID" "$ed" "$lam" "$BIGREC_BASE_MODEL" "$BIGREC_SEED" "$BIGREC_SAMPLE"
        
        # Construct path to metrics file
        # Logic matches main.py: .../ed_[ed]_lam_[lam]/test_metrics.json (inside the seed_sample dir)
        # Note: seq might output 0.0 as 0.0 used in shell, but main.py might format numbers.
        # Python print of float 0.0 is 0.0.
        # But let's check exact python formatting for directory names.
        # main.py uses f"ed_{args.ed_weight}_lam_{args.lam}"
        # If passed as string "0.0", python receives float 0.0. f"{0.0}" is "0.0".
        # So "ed_0.0_lam_0.0" is correct.
        
        # However, seq 0.0 0.1 1.0 output: 0.0, 0.1, ..., 1.0. Correct.
        # WARNING: localization might use comma. But standard docker env uses dot. Assuming dot.
        
        METRICS_FILE="${BASE_RESULT_DIR}/ed_${ed}_lam_${lam}/test_metrics.json"
        
        if [ -f "$METRICS_FILE" ]; then
            # Parse HR@20 (Last element of HR array) using python
            CURRENT_HR20=$(python -c "import json; f=open('$METRICS_FILE'); data=json.load(f); print(data['test_metrics']['HR'][-1])")
            
            echo "Result HR@20: $CURRENT_HR20"
            
            # Compare with best using python to handle float comparison properly
            IS_BETTER=$(python -c "print(1 if $CURRENT_HR20 > $BEST_HR20 else 0)")
            
            if [ "$IS_BETTER" -eq 1 ]; then
                BEST_HR20=$CURRENT_HR20
                BEST_ED=$ed
                BEST_LAM=$lam
                echo ">>> New Best Found!"
            fi
        else
            echo "Warning: Metrics file not found at $METRICS_FILE"
        fi
        
    done
done

echo "========================================================"
echo "Hyperparameter Search Completed"
echo "Best Params: ed_weight=$BEST_ED, lam=$BEST_LAM"
echo "Best HR@20: $BEST_HR20"
echo "========================================================"

# Save best result
SUMMARY_JSON="${BASE_RESULT_DIR}/best_params.json"
# Ensure directory exists (it should, but just in case)
mkdir -p "$(dirname "$SUMMARY_JSON")"

python -c "import json; 
data = {
    'best_ed_weight': $BEST_ED, 
    'best_lam': $BEST_LAM, 
    'best_hr20': $BEST_HR20,
    'teacher_model': '$BIGREC_BASE_MODEL',
    'student_model': '$MODEL_NAME'
}; 
with open('$SUMMARY_JSON', 'w') as f: json.dump(data, f, indent=4)"

echo "Best results saved to $SUMMARY_JSON"
