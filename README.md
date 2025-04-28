# DHCR Rent Registration Parser - Local Docker Version

This application parses DHCR Rent Registration PDF files, extracts data, calculates vacancy allowances, and presents the results in a web interface.

It is packaged using Docker for easy local execution.

## Prerequisites

*   **Docker Desktop**: You must have Docker Desktop installed and running on your system (Windows, macOS, or Linux).
    *   Download from: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

## !! IMPORTANT: Gemini API Key Required !!

Even though this application runs locally via Docker, the core PDF data extraction logic in `pdf_pipeline.py` **still relies on Google's Gemini API**. This means:

1.  You **need an internet connection** when processing PDFs.
2.  You **need a Gemini API key**.
    *   Get one here: [https://ai.google.dev/](https://ai.google.dev/)
3.  You must make the API key available to the application inside the Docker container.

**How to provide the API Key:**

The recommended way is to create a file named `.env` in the *same directory* as this README file and the `Dockerfile`. Add your API key to this file:

```.env
GEMINI_API_KEY=YOUR_API_KEY_HERE
```

Replace `YOUR_API_KEY_HERE` with your actual key.

Then, you need to slightly modify the `docker run` command in `run.sh` and `run.bat` to mount this file. Uncomment and adjust the lines that look like this:

*   In `run.sh`:
    ```bash
    # Optional: Mount a local .env file (replace /path/to/your/.env with actual path)
    # docker run --rm -d -p 8080:8080 --name $CONTAINER_NAME --env-file /path/to/your/.env $IMAGE_NAME
    # MODIFY TO:
    docker run --rm -d -p 8080:8080 --name $CONTAINER_NAME --env-file .env $IMAGE_NAME
    ```
*   In `run.bat`:
    ```bat
    REM Optional: Mount a local .env file (replace C:\path\to\your\.env with actual path)
    REM docker run --rm -d -p 8080:8080 --name %CONTAINER_NAME% --env-file C:\path\to\your\.env %IMAGE_NAME%
    REM MODIFY TO:
    docker run --rm -d -p 8080:8080 --name %CONTAINER_NAME% --env-file .env %IMAGE_NAME%
    ```

**(Do not commit your `.env` file to version control if you use Git!)**

## How to Run

1.  **Open a Terminal or Command Prompt** in the directory containing this README file and the other scripts (`Dockerfile`, `run.sh`, `run.bat`, etc.).
2.  **Build and Start the Application:**
    *   **Linux/macOS**: Run `./run.sh`
    *   **Windows**: Run `run.bat`

    This script will first build the Docker image (which might take a few minutes the first time) and then start the application container in the background.
3.  **Access the Application**: Open your web browser and go to:
    [http://localhost:8080](http://localhost:8080)

## How to Stop

1.  **Open a Terminal or Command Prompt** in the same directory.
2.  **Stop the Application:**
    *   **Linux/macOS**: Run `./stop.sh`
    *   **Windows**: Run `stop.bat`

    This will stop the running Docker container. Because it was started with `--rm`, Docker will also automatically remove the container, cleaning up its resources.

## Temporary Files

The application uses temporary storage *inside* the Docker container (in `/tmp`) for uploads and processed files. This storage is ephemeral:

*   Stopping the container (via `stop.sh` or `stop.bat`) effectively removes these temporary files because the container is removed.
*   The application also has a "Reset" button in the UI which clears these internal temporary directories. 