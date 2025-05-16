# DHCR and CBB Parser - Local Version

This application parses DHCR Rent Registration PDF files, scans and extracts data with AI vision models, calculates vacancy allowances, and compiles and exports the parsed data for Excel. It also presents the scan and calculation results in a web interface.

## Prerequisites

## !! IMPORTANT: Gemini API Key Required !!

Even though this application runs locally via Docker, the core PDF data extraction logic **still relies on Google's Gemini API**. This means:

1.  You **need an internet connection** when processing PDFs.
2.  You **need a Gemini API key**.
    *   Get one for free here: [https://ai.google.dev/](https://ai.google.dev/)
3.  Add your GEMINI API Ke to the config file with a text editor

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
        *   Build and download the application files image (this might take a few minutes).
        *   Start the application container and open the app in your default browser.
    *   Subsequent runs will just start the container (rebuilding only if necessary).
4.  **Access the Application**:
    *   Open your web browser and go to: [http://127.0.0.1:7865] (the final 4 digits may shift if that address is unavilable on your machine)
5.  **Stop the Application**:
    While this should not be needed, since the app is designed to clean up after iteslf, here is how to close instances of the app:
    *   Right-click the `rgbo_main` folder (the folder `config` was inside of) and select `New Terminal at Folder`
    *   Run the command: `bash stop.sh`
    *   This will stop the instances of the app.

## Windows Usage


!!!~~~ WIP ~~~!!!
