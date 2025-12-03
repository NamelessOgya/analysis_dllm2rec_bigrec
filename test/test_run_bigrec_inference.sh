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

# Create mock python
cat << 'EOF' > "$WORKSPACE/mock_python"
#!/bin/bash
echo "Mock Python called with: $@"

# Simulate inference output
if [[ "$@" == *"inference.py"* ]]; then
    # Extract output path
    # args: ... --result_json_data path ...
    OUTPUT_FILE=""
    prev=""
    for arg in "$@"; do
        if [[ "$prev" == "--result_json_data" ]]; then
            OUTPUT_FILE="$arg"
        fi
        prev="$arg"
    done
    
    if [ -n "$OUTPUT_FILE" ]; then
        echo "Creating dummy inference output at $OUTPUT_FILE"
        mkdir -p "$(dirname "$OUTPUT_FILE")"
        echo '[]' > "$OUTPUT_FILE"
    fi
fi

# Simulate evaluate output
if [[ "$@" == *"evaluate.py"* ]]; then
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
echo "Executing: ./cmd/run_bigrec_inference.sh movie 0 \"Qwen/Qwen2-0.5B\" 0 1024"
./cmd/run_bigrec_inference.sh movie 0 "Qwen/Qwen2-0.5B" 0 1024

# Verify results
EXPECTED_RESULT_DIR="BIGRec/results/movie/Qwen_Qwen2-0.5B/0_1024"
if [ -f "$EXPECTED_RESULT_DIR/test.json" ]; then
    echo "SUCCESS: Inference result found at $EXPECTED_RESULT_DIR/test.json"
else
    echo "FAILURE: Inference result NOT found at $EXPECTED_RESULT_DIR/test.json"
    exit 1
fi

if [ -f "$EXPECTED_RESULT_DIR/metrics.json" ]; then
    echo "SUCCESS: Evaluation metrics found at $EXPECTED_RESULT_DIR/metrics.json"
else
    echo "FAILURE: Evaluation metrics NOT found at $EXPECTED_RESULT_DIR/metrics.json"
    exit 1
fi

echo "Test Passed!"
