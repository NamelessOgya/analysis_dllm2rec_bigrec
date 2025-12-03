#!/bin/bash

# Container name
CONTAINER_NAME="dllm2rec_bigrec_container"
IMAGE_NAME="dllm2rec-bigrec:latest"

# Check if container exists
if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Container $CONTAINER_NAME already exists."
    if [ "$(docker ps -aq -f status=exited -f name=^/${CONTAINER_NAME}$)" ]; then
        echo "Starting existing container..."
        docker start $CONTAINER_NAME
    else
        echo "Container is already running."
    fi
else
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
        -v $(pwd):/workspace \
        $IMAGE_NAME
fi

echo "Container started. You can attach to it using:"
echo "docker exec -it $CONTAINER_NAME /bin/bash"
