#!/bin/bash
set -e

# Default to pipeline_params.csv if no argument provided
CSV_PATH="${1:-pipeline_params.csv}"

# Check if file exists
if [ ! -f "$CSV_PATH" ]; then
    echo "Error: Configuration file '$CSV_PATH' not found."
    echo "Usage: $0 [path_to_csv]"
    exit 1
fi

echo "Generating DVC pipeline from $CSV_PATH..."
python3 cmd/generate_dvc_pipeline.py "$CSV_PATH"
echo "Done."
