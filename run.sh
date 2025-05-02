#!/bin/bash

# Script to set up the environment, build, and run the Docker container for the DHCR Parser app

# --- Start Setup Block ---
ENV_FILE=".env"
GEMINI_API_KEY_PRESENT=false

# Function to prompt for API key and update .env
prompt_and_update_env() {
    echo ""
    echo "---------------------------------------------------------------------"
    echo "You **need a Gemini API key** to run the analysis features."
    echo "Get one for free at https://ai.google.dev/"
    echo "---------------------------------------------------------------------"
    echo ""
    read -p "Please paste your Gemini API Key and press Enter: " GEMINI_API_KEY_INPUT

    # Validate if the key looks potentially empty (basic check)
    if [ -z "$GEMINI_API_KEY_INPUT" ]; then
        echo "Warning: No API key entered. Analysis features requiring the key will fail."
        # Decide if you want to exit or continue without a key
        # exit 1 # Uncomment to exit if key is mandatory
    fi

    echo "Updating $ENV_FILE..."
    # Overwrite or create the file with the new content
    cat << EOF > "$ENV_FILE"
GEMINI_API_KEY=${GEMINI_API_KEY_INPUT}
GEMINI_MODEL="gemini-1.5-flash-latest"
EOF
    GEMINI_API_KEY_PRESENT=true
    echo "$ENV_FILE has been updated."
    echo "---------------------------------------------------------------------"
    echo ""
}

# 1. Check if .env file exists and contains the API key
if [ -f "$ENV_FILE" ]; then
    echo "$ENV_FILE found."
    # Check if GEMINI_API_KEY is present and non-empty within the file
    if grep -q "^GEMINI_API_KEY=.\+" "$ENV_FILE"; then
        echo "Gemini API Key found in $ENV_FILE."
        GEMINI_API_KEY_PRESENT=true
    else
        echo "Gemini API Key not found or is empty in $ENV_FILE."
        prompt_and_update_env
    fi
else
    echo "$ENV_FILE not found."
    prompt_and_update_env
fi

# Optional: Exit if API key is mandatory and wasn't provided
# if [ "$GEMINI_API_KEY_PRESENT" = false ]; then
#     echo "Error: Gemini API Key is required but was not provided. Exiting."
#     exit 1
# fi
# --- End Setup Block ---

# Script to build and run the Docker container for the DHCR Parser app

IMAGE_NAME="dhcr-parser-app"
CONTAINER_NAME="dhcr-parser-container"

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo "Docker build failed. Exiting."
    exit 1
fi

echo "Build successful."

# Check if a container with the same name is already running
if [ $(docker ps -q -f name=^/${CONTAINER_NAME}$) ]; then
    echo "Container $CONTAINER_NAME is already running. Stopping and removing it..."
    docker stop $CONTAINER_NAME
    # No need to remove if using --rm on run
fi

# Check if a container with the same name exists (but is stopped)
if [ $(docker ps -aq -f status=exited -f name=^/${CONTAINER_NAME}$) ]; then
    echo "Removing existing stopped container $CONTAINER_NAME..."
    docker rm $CONTAINER_NAME
fi

# Run the Docker container
echo "Running Docker container: $CONTAINER_NAME..."
echo "Access the application at http://localhost:8080"

# Run in detached mode (-d)
# Automatically remove the container when it exits (--rm)
# Map port 8080 on the host to port 8080 in the container (-p 8080:8080)
# Give the container a name (--name)
# Optional: Mount a local .env file (replace /path/to/your/.env with actual path)
# docker run --rm -d -p 8080:8080 --name $CONTAINER_NAME --env-file /path/to/your/.env $IMAGE_NAME
# Use the .env file in the current directory
docker run --rm -d -p 8080:8080 --name $CONTAINER_NAME --env-file .env $IMAGE_NAME

if [ $? -ne 0 ]; then
    echo "Failed to start Docker container $CONTAINER_NAME. Check Docker logs."
    exit 1
fi

echo "Container $CONTAINER_NAME started successfully."

# Attempt to open the browser
echo "Attempting to open the application in your default browser..."
if [[ "$(uname)" == "Darwin" ]]; then
  open http://localhost:8080
elif [[ "$(uname)" == "Linux" ]]; then
  xdg-open http://localhost:8080
else
  echo "Could not automatically open browser. Please navigate to http://localhost:8080 manually."
fi

exit 0 # Exit cleanly after potentially opening browser 