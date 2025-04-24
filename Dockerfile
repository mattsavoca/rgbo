# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies - including poppler-utils
# Use apt-get as the base image is Debian-based
RUN apt-get update && apt-get install -y --no-install-recommends \\\
    poppler-utils \\\
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on (default for serve() is 5001, but Vercel expects 80 or 443 often)
# Uvicorn needs to bind to 0.0.0.0 to be accessible from outside the container.
# Vercel will map its external port to this internal port. Let's use 8000 as a common practice.
EXPOSE 8000

# Command to run the application using uvicorn
# Match the host and port FastHTML's serve() might try to use, or override.
# We need to explicitly run uvicorn here instead of relying on serve().
# 'app:app' refers to the 'app' instance inside the 'app.py' file.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 