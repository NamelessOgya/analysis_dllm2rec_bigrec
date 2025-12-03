#!/bin/bash

# Check if a token is provided as an argument
if [ -n "$1" ]; then
    echo "Logging in with provided token..."
    huggingface-cli login --token "$1"
else
    echo "No token provided. Starting interactive login..."
    echo "Please paste your Hugging Face token when prompted."
    huggingface-cli login
fi
