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
COPY gradio_app.py .
COPY scripts/ ./scripts/
COPY tabs/ ./tabs/
COPY rgbo.csv .
COPY footnotes.csv .
# Copy .env file if you want to bundle API keys (NOT RECOMMENDED FOR SHARING)
# Better to mount or use environment variables in 'docker run'
# COPY .env .

# Make port 8080 available to the world outside this container
# (Using 8080 based on typical Cloud Run/containerized app convention and your main.py fallback)
EXPOSE 8080

# Define environment variable for the port (aligns with main.py)
# Gradio uses 7860 by default, but Cloud Run expects 8080 unless configured otherwise.
# We'll need to tell Gradio to use this port when launching.
# However, Gradio's demo.launch() doesn't easily take PORT env var directly for the server port.
# It's simpler to run it directly and let Cloud Run map the default Gradio port (7860) or configure Cloud Run service port.
# Sticking with 8080 for EXPOSE as it's common convention. The CMD will use Gradio's default.
# Or we can modify gradio_app.py to read the PORT env var.
# For now, let's keep EXPOSE 8080 and run Gradio directly.
# ENV PORT=8080 # Keep this if gradio_app.py is modified to use it

# Run gradio_app.py when the container launches
# Using python directly is standard for Gradio apps.
CMD ["python", "gradio_app.py"] 