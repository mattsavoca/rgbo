#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Running build.sh..."

# Update package list and install Poppler utilities
# Vercel uses Amazon Linux 2, which uses yum
echo "Updating yum and installing poppler-utils..."
yum update -y
yum install -y poppler-utils

echo "Poppler should now be installed."

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Create a dummy output directory for the static build step
# Vercel requires the build step to produce an output directory
mkdir -p static_build_output
touch static_build_output/dummy.txt

echo "build.sh finished." 