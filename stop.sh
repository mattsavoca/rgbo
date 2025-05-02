#!/bin/bash

# Script to stop the DHCR Parser app Docker container

CONTAINER_NAME="dhcr-parser-container"

# Check if the container is running
if [ $(docker ps -q -f name=^/${CONTAINER_NAME}$) ]; then
    echo "Stopping container $CONTAINER_NAME..."
    docker stop $CONTAINER_NAME
    if [ $? -eq 0 ]; then
        echo "Container $CONTAINER_NAME stopped successfully."
        echo "(Container was run with --rm, so it should be removed automatically)."
    else
        echo "Failed to stop container $CONTAINER_NAME. Check Docker status."
    fi
else
    echo "Container $CONTAINER_NAME is not currently running."
fi 