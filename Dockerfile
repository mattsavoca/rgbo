# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies - including poppler-utils
# Use apt-get as the base image is Debian-based
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for the BACKEND
# Copy only requirements_backend.txt first to leverage Docker cache
COPY requirements_backend.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_backend.txt

# Copy the rest of the application code needed for the backend
# Includes app_backend.py, pdf_handler.py, pdf_pipeline.py, calculate_vacancy_allowance.py, pdf_to_png.py, rgbo.csv
# Exclude frontend app.py and its requirements.txt if they are in the same dir during build context
COPY app_backend.py .
COPY pdf_handler.py .
COPY pdf_pipeline.py .
COPY calculate_vacancy_allowance.py .
COPY pdf_to_png.py .
COPY rgbo.csv . 
# If other necessary data files exist, copy them too.
# Consider creating a .dockerignore file to exclude unnecessary files (like .git, frontend files, large PDFs)

# Expose the port the app runs on
EXPOSE 8000

# Command to run the backend application using uvicorn
# 'app_backend:app' refers to the 'app' instance inside the 'app_backend.py' file.
CMD ["uvicorn", "app_backend:app", "--host", "0.0.0.0", "--port", "8000"] 