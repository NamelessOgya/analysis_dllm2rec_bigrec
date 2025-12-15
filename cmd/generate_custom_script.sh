#!/bin/bash
# Wrapper script to run the python pipeline generator
# Usage: ./cmd/generate_custom_script.sh [csv_path]

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3 "$DIR/generate_custom_script.py" "$@"
