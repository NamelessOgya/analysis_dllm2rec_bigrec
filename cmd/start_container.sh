#!/bin/bash

# Container name
CONTAINER_NAME="dllm2rec_bigrec_container"
IMAGE_NAME="dllm2rec-bigrec:latest"

# Check if container exists and remove it
if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Removing existing container $CONTAINER_NAME..."
    docker rm -f $CONTAINER_NAME
fi

echo "Creating and starting new container..."
# Mount current directory to /workspace
# Use --gpus all if GPU is available, otherwise comment it out or handle dynamically
# For now, we assume GPU might not be available on this machine (as per user request), 
# but we include the flag commented out or check for nvidia-smi.

GPU_FLAG=""
if command -v nvidia-smi &> /dev/null; then
    GPU_FLAG="--gpus all"
fi

docker run -d -it \
    --name $CONTAINER_NAME \
    $GPU_FLAG \
    --shm-size=8g \
    --memory=16g \
    -v $(pwd):/workspace \
    $IMAGE_NAME

echo "Container started. You can attach to it using:"
echo "docker exec -it $CONTAINER_NAME /bin/bash"
