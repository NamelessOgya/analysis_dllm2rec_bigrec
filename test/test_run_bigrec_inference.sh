#!/bin/bash
set -e

# Create a temporary workspace
WORKSPACE=$(mktemp -d)
echo "Test workspace: $WORKSPACE"

# Setup directory structure
mkdir -p "$WORKSPACE/BIGRec/data/movie"
mkdir -p "$WORKSPACE/BIGRec/model/movie/Qwen_Qwen2-0.5B/0_1024" # Mock LoRA weights
mkdir -p "$WORKSPACE/cmd"

# Copy the script to be tested
# Assuming we run this test from the project root
cp "$(pwd)/cmd/run_bigrec_inference.sh" "$WORKSPACE/cmd/"

# Create dummy inference.py and evaluate.py
touch "$WORKSPACE/BIGRec/inference.py"
touch "$WORKSPACE/BIGRec/data/movie/evaluate.py"

# Create dummy test data
mkdir -p "$WORKSPACE/BIGRec/data/movie"
echo '[{"instruction": "test", "input": "test", "output": "test"}]' > "$WORKSPACE/BIGRec/data/movie/test_5000.json"

# Create mock python
cat << 'EOF' > "$WORKSPACE/mock_python"
#!/bin/bash
echo "Mock Python called with: $@"

# Simulate inference output
if [[ "$@" == *"inference.py"* ]]; then
    # Extract result json path
    RESULT_JSON=""
    # Make a copy of arguments to parse without affecting the original "$@"
    ARGS=("$@")
    while [[ ${#ARGS[@]} -gt 0 ]]; do
        if [[ "${ARGS[0]}" == "--result_json_data" ]]; then
            RESULT_JSON="${ARGS[1]}"
            ARGS=("${ARGS[@]:2}") # Shift 2
        else
            ARGS=("${ARGS[@]:1}") # Shift 1
        fi
    done
    
    if [ -z "$RESULT_JSON" ]; then
        # Fallback if not found (should not happen with current script)
        RESULT_JSON="test.json"
    fi

    echo "Creating dummy inference output at $RESULT_JSON"
    mkdir -p "$(dirname "$RESULT_JSON")"
    echo '[{"predict": ["test"], "output": "test"}]' > "$RESULT_JSON"
fi

# Simulate evaluate output
if [[ "$@" == *"evaluate.py"* ]]; then
    # If --save_results is passed, create dummy rank and score files
    SAVE_RESULTS=false
    for arg in "$@"; do
        if [[ "$arg" == "--save_results" ]]; then
            SAVE_RESULTS=true
            break
        fi
    done
    
    echo "Mock evaluate.py: SAVE_RESULTS=$SAVE_RESULTS"

    if [ "$SAVE_RESULTS" = true ]; then
        # The real script iterates over files in input_dir.
        # Here we assume input_dir contains the result json we just created.
        # We need to find the input_dir argument.
        INPUT_DIR=""
        ARGS=("$@")
        while [[ ${#ARGS[@]} -gt 0 ]]; do
            if [[ "${ARGS[0]}" == "--input_dir" ]]; then
                INPUT_DIR="${ARGS[1]}"
                break
            fi
            ARGS=("${ARGS[@]:1}")
        done
        
        echo "Mock evaluate.py: INPUT_DIR=$INPUT_DIR"
        
        if [ -n "$INPUT_DIR" ]; then
            # Create dummy files for any json in input dir
            # Note: The glob might fail if no files, so check existence
            shopt -s nullglob
            for json_file in "$INPUT_DIR"/*.json; do
                echo "Mock evaluate.py: Found json file $json_file"
                if [ -f "$json_file" ]; then
                    base_name="${json_file%.*}"
                    echo "Creating dummy rank/score for $base_name"
                    touch "${base_name}_rank.txt"
                    touch "${base_name}_score.txt"
                fi
            done
            shopt -u nullglob
        fi
    fi
    # Check for base_model argument
    BASE_MODEL_ARG=""
    prev=""
    for arg in "$@"; do
        if [[ "$prev" == "--base_model" ]]; then
            BASE_MODEL_ARG="$arg"
        fi
        prev="$arg"
    done

    if [ -z "$BASE_MODEL_ARG" ]; then
        echo "Error: --base_model argument missing in evaluate.py call"
        exit 1
    fi
    echo "evaluate.py called with base_model: $BASE_MODEL_ARG"

    # evaluate.py writes to ./movie.json (based on dataset name usually)
    # The script expects ./movie.json (or whatever dataset name is)
    # We need to know the dataset name. It's not passed to evaluate.py directly as arg in the script call 
    # (python "./data/$DATASET/evaluate.py" --input_dir "$RESULT_DIR")
    # But the script we are testing expects ./movie.json to exist after this call.
    # So we just create it.
    echo "Creating dummy evaluation output ./movie.json"
    echo '{}' > "./movie.json"
fi
EOF
chmod +x "$WORKSPACE/mock_python"

# Run the test
cd "$WORKSPACE"
export PATH="$WORKSPACE:$PATH"

# Link mock python
ln -s "$WORKSPACE/mock_python" "$WORKSPACE/python"

# Execute the script
echo "Executing: ./cmd/run_bigrec_inference.sh movie 0 \"Qwen/Qwen2-0.5B\" 0 1024 false test_5000.json"
# Create dummy embedding file to avoid generation step in test
mkdir -p "BIGRec/data/movie/model_embeddings"
touch "BIGRec/data/movie/model_embeddings/Qwen_Qwen2-0.5B.pt"

./cmd/run_bigrec_inference.sh movie 0 "Qwen/Qwen2-0.5B" 0 1024 false test_5000.json

# Verify results
EXPECTED_RESULT_DIR="BIGRec/results/movie/Qwen_Qwen2-0.5B/0_1024"
if [ -f "$EXPECTED_RESULT_DIR/test_5000.json" ]; then
    echo "SUCCESS: Inference result found at $EXPECTED_RESULT_DIR/test_5000.json"
else
    echo "FAILURE: Inference result NOT found at $EXPECTED_RESULT_DIR/test_5000.json"
    exit 1
fi

if [ -f "$EXPECTED_RESULT_DIR/metrics.json" ]; then
    echo "SUCCESS: Evaluation metrics found at $EXPECTED_RESULT_DIR/metrics.json"
else
    echo "FAILURE: Evaluation metrics NOT found at $EXPECTED_RESULT_DIR/metrics.json"
    exit 1
fi

# Verify rank and score files
if [ -f "$EXPECTED_RESULT_DIR/test_5000_rank.txt" ]; then
    echo "SUCCESS: Rank file found at $EXPECTED_RESULT_DIR/test_5000_rank.txt"
else
    echo "FAILURE: Rank file NOT found at $EXPECTED_RESULT_DIR/test_5000_rank.txt"
    exit 1
fi

if [ -f "$EXPECTED_RESULT_DIR/test_5000_score.txt" ]; then
    echo "SUCCESS: Score file found at $EXPECTED_RESULT_DIR/test_5000_score.txt"
else
    echo "FAILURE: Score file NOT found at $EXPECTED_RESULT_DIR/test_5000_score.txt"
    exit 1
fi

echo "Test Passed!"

# Test Case 2: Skip Inference
echo "Executing Test Case 2: Skip Inference"
# Remove previous results
rm -rf "$WORKSPACE/BIGRec/results"
# Re-create dummy inference output manually because inference.py will be skipped but evaluate.py needs it
mkdir -p "$WORKSPACE/$EXPECTED_RESULT_DIR"
echo '[]' > "$WORKSPACE/$EXPECTED_RESULT_DIR/test_5000.json"

./cmd/run_bigrec_inference.sh movie 0 "Qwen/Qwen2-0.5B" 0 1024 true test_5000.json

# Verify results
if [ -f "$EXPECTED_RESULT_DIR/metrics.json" ]; then
    echo "SUCCESS: Evaluation metrics found at $EXPECTED_RESULT_DIR/metrics.json (Skip Inference)"
else
    echo "FAILURE: Evaluation metrics NOT found at $EXPECTED_RESULT_DIR/metrics.json (Skip Inference)"
    exit 1
fi

echo "Test Case 2 Passed!"
