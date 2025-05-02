# DHCR Rent Registration Parser - Local Docker Version

This application parses DHCR Rent Registration PDF files, extracts data, calculates vacancy allowances, and presents the results in a web interface.

It is packaged using Docker for easy local execution.

## Prerequisites

*   **Docker Desktop**: You must have Docker Desktop installed and running on your system (Windows, macOS, or Linux).
    *   Download from: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

## !! IMPORTANT: Gemini API Key Required !!

Even though this application runs locally via Docker, the core PDF data extraction logic **still relies on Google's Gemini API**. This means:

1.  You **need an internet connection** when processing PDFs.
2.  You **need a Gemini API key**.
    *   Get one for free here: [https://ai.google.dev/](https://ai.google.dev/)

**How the API Key is Handled:**

*   When you run the application for the first time using the `run.sh` (Mac/Linux) or `run.bat` (Windows) script, it will **automatically check** if a `.env` file containing your API key exists in the same directory.
*   If the file or the key is missing, the script will **prompt you** to paste your Gemini API Key into `config`
*   The script will then create or update the `.env` file for you with the key you provide.
*   Subsequent runs will use the key stored in the `.env` file without prompting you again.

**Note to Developers: Never commit a `.env` file to version control platforms. It contains your secret API key.)**

## Mac/Linux Usage

1.  **Unzip dhcr_main.zip** to a folder of your choice
2. Open the `rgbo-main` folder and open `config`
3. Replace `PLACE API KEY HERE` with your own API key from Google, save, close `config` and close the finder window.
4. Right-click the `rgbo_main` folder (the folder `config` was inside of) and select `New Terminal at Folder`
5.  **Start the Application**: Run the command: `bash run.sh`
    *   The first time you run this, it will:
        *   Prompt you for your Gemini API Key if it hasn't been saved yet.
        *   Build the Docker image (this might take a few minutes).
        *   Start the application container.
    *   Subsequent runs will just start the container (rebuilding only if necessary).
4.  **Access the Application**:
    *   Open your web browser and go to: [http://localhost:8080](http://localhost:8080)
5.  **Stop the Application**:
    While this should not be needed, since the app is designed to clean up after iteslf, here is how to close instances of the app:
    *   Right-click the `rgbo_main` folder (the folder `config` was inside of) and select `New Terminal at Folder`
    *   Run the command: `bash stop.sh`
    *   This will stop the instances of the app.

## Windows Usage



!!!~~~ WIP ~~~!!!
1.  **Open a Command Prompt or PowerShell** in the directory containing this README file and the other scripts (`Dockerfile`, `run.bat`, `stop.bat`, etc.).
2.  **Start the Application**:
    *   Run the command: `run.bat`
    *   The first time you run this, it will:
        *   Prompt you for your Gemini API Key if it hasn't been saved yet.
        *   Build the Docker image (this might take a few minutes).
        *   Start the application container.
    *   Subsequent runs will just start the container (rebuilding only if necessary).
3.  **Access the Application**:
    *   Open your web browser and go to: [http://127.0.0.1:7865] (the final 4 digits may shift if that address is unavilable on your machine)
4.  **Manually Stop the Application**:
    *   Open a Command Prompt or PowerShell in the same directory.
    *   Run the command: `stop.bat`
    *   This will stop and remove the running Docker container.

## Temporary Files

The application uses temporary storage *inside* the Docker container (in `/tmp`) for uploads and processed files. This storage is ephemeral:

*   Stopping the container (via `stop.sh` or `stop.bat`) effectively removes these temporary files because the container is removed.
*   The application also has a "Reset" button in the UI which clears these internal temporary directories. 
