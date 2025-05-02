# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (PyMuPDF/fitz usually doesn't require extra ones like Poppler)
# RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils && rm -rf /var/lib/apt/lists/* # Example if poppler-utils were needed

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# Ensure ALL necessary .py files and data files (like rgbo.csv) are copied
COPY main.py .
COPY pdf_handler.py .
COPY pdf_pipeline.py .
COPY pdf_to_png.py .
COPY calculate_vacancy_allowance.py .
COPY rgbo.csv .
# Copy .env file if you want to bundle API keys (NOT RECOMMENDED FOR SHARING)
# Better to mount or use environment variables in 'docker run'
# COPY .env .

# Make port 8080 available to the world outside this container
# (Using 8080 based on typical Cloud Run/containerized app convention and your main.py fallback)
EXPOSE 8080

# Define environment variable for the port (aligns with main.py)
ENV PORT=8080

# Run main.py when the container launches
# The command uses uvicorn directly, which is common for ASGI apps like FastAPI/Starlette/FastHTML
# If your serve() function in main.py handles host/port correctly, this might also work:
# CMD ["python", "main.py"]
# Using uvicorn provides more control and is standard practice:
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 