# Use an official Python image. Using bookworm allows easy install of poppler
FROM python:3.12-slim-bookworm

# Install poppler-utils (needed by pdf2image) and cleanup apt cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the consolidated requirements file
COPY requirements.txt .

# Install Python dependencies
# Consider using --no-cache-dir to potentially reduce final image size slightly
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code and necessary data files
COPY main.py .
COPY pdf_handler.py .
COPY pdf_pipeline.py .
COPY calculate_vacancy_allowance.py .
COPY pdf_to_png.py .
COPY rgbo.csv .

# Create the directories needed by the application within the container
RUN mkdir temp_uploads processed_data

# Make port 8080 available (standard for Cloud Run)
EXPOSE 8080

# Define environment variable defaults (Cloud Run will override PORT)
ENV PORT=8080
ENV HOST=0.0.0.0
# GEMINI_API_KEY will be set via Cloud Run service environment variables/secrets

# Run uvicorn when the container launches, pointing to the merged app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]