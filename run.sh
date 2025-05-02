#!/bin/bash

# Script to set up environment and run the DHCR Parser Gradio app locally

# --- Configuration ---
CONFIG_FILE="config"
ENV_FILE=".env"
VENV_DIR="venv"
REQUIRED_PYTHON_VERSION="3.11" # Minimum required version
GRADIO_DEFAULT_URL="http://127.0.0.1:7860" # Default Gradio URL

echo "--- DHCR Parser Local Runner ---"
echo "INFO: This script should be run WITHOUT admin/sudo privileges."

# --- Helper Functions ---
check_python_version() {
    local python_cmd=$1
    if ! command -v $python_cmd &> /dev/null; then
        return 1 # Command not found
    fi

    # Get version X.Y
    local version_str=$($python_cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
    if [ -z "$version_str" ]; then
        echo "WARN: Could not determine Python version for $python_cmd."
        return 1 # Version check failed
    fi

    echo "INFO: Found Python ($python_cmd) version: $version_str"

    # Version comparison (requires sort -V)
    if [[ "$(printf '%s\n' "$REQUIRED_PYTHON_VERSION" "$version_str" | sort -V | head -n1)" != "$REQUIRED_PYTHON_VERSION" ]]; then
        echo "WARN: Python version $version_str is older than required $REQUIRED_PYTHON_VERSION."
        return 1 # Version too old
    fi
    return 0 # Version is OK
}

# --- Main Script ---

# 1. Check for config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file '$CONFIG_FILE' not found."
    echo "Please ensure '$CONFIG_FILE' exists in the same directory as this script."
    # You could create a default one here if you prefer:
    # echo "INFO: Creating a default '$CONFIG_FILE'. Please edit it with your API key."
    # cat << EOF > "$CONFIG_FILE"
# GEMINI_API_KEY=PLACEHOLDER
# GEMINI_MODEL="gemini-1.5-flash-latest"
# EOF
    exit 1
fi
echo "INFO: Found configuration file: $CONFIG_FILE"

# 2. Check API Key Placeholder in config
# Using grep -F to treat the string literally, just in case
if grep -q -F "GEMINI_API_KEY=PLACEHOLDER" "$CONFIG_FILE"; then
    echo "---------------------------------------------------------------------"
    echo "ACTION REQUIRED:"
    echo "Please open the '$CONFIG_FILE' file and replace 'PLACEHOLDER'"
    echo "with your actual Google AI Gemini API Key."
    echo "Get one for free at https://ai.google.dev/"
    echo "---------------------------------------------------------------------"
    echo "Re-run this script after updating the file."
    exit 1
fi
echo "INFO: Gemini API Key seems to be set in $CONFIG_FILE."

# 3. Find suitable Python command (python3 or python)
PYTHON_CMD=""
if check_python_version "python3"; then
    PYTHON_CMD="python3"
elif check_python_version "python"; then
    PYTHON_CMD="python"
else
    # Check if on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "INFO: No suitable Python $REQUIRED_PYTHON_VERSION+ found on macOS. Attempting to install/update via Homebrew..."

        # 1. Check for Homebrew
        if ! command -v brew &> /dev/null; then
            echo "INFO: Homebrew not found. Attempting to install Homebrew..."
            # Run the official Homebrew installer script
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            if [ $? -ne 0 ]; then
                echo "---------------------------------------------------------------------"
                echo "ERROR: Homebrew installation failed."
                echo "Please try installing Homebrew manually:"
                echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                echo "And then re-run this script ($0)."
                echo "---------------------------------------------------------------------"
                exit 1
            fi
            echo "INFO: Homebrew installed successfully."
            # NOTE: Homebrew might add itself to the PATH in shell config files,
            # but it might not be available *immediately* in this script's session.
            # We proceed hoping it's available or the user restarts the terminal if needed.
            # Adding brew to PATH explicitly here is tricky and varies.
        else
            echo "INFO: Homebrew is already installed."
        fi

        # 2. Install/Update Python using Homebrew
        echo "INFO: Attempting to install/update Python via Homebrew (brew install python)..."
        brew install python
        if [ $? -ne 0 ]; then
            echo "---------------------------------------------------------------------"
            echo "ERROR: Failed to install Python using 'brew install python'."
            echo "Please try running 'brew update && brew install python' manually,"
            echo "ensure Python $REQUIRED_PYTHON_VERSION+ is installed and in your PATH,"
            echo "and then re-run this script ($0)."
            echo "---------------------------------------------------------------------"
            exit 1
        fi
        echo "INFO: Python installed/updated via Homebrew."

        # 3. Re-check for Python command after installation attempt
        echo "INFO: Re-checking for Python command..."
        if check_python_version "python3"; then
            PYTHON_CMD="python3"
        # It's less likely plain 'python' would be the Homebrew one, but check anyway
        elif check_python_version "python"; then
             PYTHON_CMD="python"
        else
             echo "---------------------------------------------------------------------"
             echo "ERROR: Python installed via Homebrew, but still couldn't find a suitable command."
             echo "Please ensure Homebrew's Python is in your PATH."
             echo "You may need to restart your terminal or follow instructions from 'brew info python'."
             echo "Re-run this script ($0) once Python $REQUIRED_PYTHON_VERSION+ is accessible."
             echo "---------------------------------------------------------------------"
             exit 1
        fi

    else
        # Original error for non-macOS systems
        echo "---------------------------------------------------------------------"
        echo "ERROR: No suitable Python installation found."
        echo "Please install Python $REQUIRED_PYTHON_VERSION or later and ensure 'python3' or 'python'"
        echo "is available in your system's PATH."
        echo "Download from: https://www.python.org/"
        echo "---------------------------------------------------------------------"
        exit 1
    fi
fi
echo "INFO: Using Python command: $PYTHON_CMD"


# 4. Determine Platform Specific Paths for venv executables
PYTHON_EXEC=""
PIP_EXEC=""
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    PYTHON_EXEC="$VENV_DIR/bin/python"
    PIP_EXEC="$VENV_DIR/bin/pip"
elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" ]]; then # Git Bash, Cygwin
    PYTHON_EXEC="$VENV_DIR/Scripts/python.exe"
    PIP_EXEC="$VENV_DIR/Scripts/pip.exe"
# elif [[ "$OSTYPE" == "win32" ]]; then # Native Windows - .sh script won't run here easily
    # This block is unlikely to be hit by run.sh but included for context
    # PYTHON_EXEC="$VENV_DIR\Scripts\python.exe"
    # PIP_EXEC="$VENV_DIR\Scripts\pip.exe"
else
    echo "ERROR: Unsupported operating system '$OSTYPE'. Cannot determine venv paths."
    echo "This script primarily supports Linux, macOS, and Windows (via Git Bash/WSL)."
    exit 1
fi


# 5. Check/Create Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    echo "INFO: Creating Python virtual environment in './$VENV_DIR'..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment using '$PYTHON_CMD -m venv $VENV_DIR'."
        exit 1
    fi
    echo "INFO: Virtual environment created."
else
    echo "INFO: Virtual environment './$VENV_DIR' already exists."
fi

# 6. Install/Upgrade Requirements into venv using the venv's pip
echo "INFO: Installing/upgrading dependencies from requirements.txt..."
"$PYTHON_EXEC" -m pip install --upgrade pip # Upgrade pip in venv
if [ $? -ne 0 ]; then
    echo "WARN: Failed to upgrade pip in the virtual environment. Continuing installation..."
fi

"$PIP_EXEC" install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies from requirements.txt using '$PIP_EXEC'."
    echo "Check your internet connection and the contents of requirements.txt."
    exit 1
fi
echo "INFO: Dependencies installed successfully."

# 7. Create .env file from config
echo "INFO: Creating/updating $ENV_FILE from $CONFIG_FILE..."
cp "$CONFIG_FILE" "$ENV_FILE"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to copy $CONFIG_FILE to $ENV_FILE."
    exit 1
fi
echo "INFO: $ENV_FILE created/updated."

# 8. Run Gradio App using venv python
echo "---------------------------------------------------------------------"
echo "INFO: Starting the Gradio application..."
echo "      Access it in your browser, usually at: $GRADIO_DEFAULT_URL"
echo "      (Check terminal output below for the exact URL if different)"
echo "      Press CTRL+C in this terminal to stop the application."
echo "---------------------------------------------------------------------"

# Run in foreground so user sees output and can Ctrl+C
"$PYTHON_EXEC" gradio_app.py &
GRADIO_PID=$! # Get PID of background process

# Give Gradio a moment to start up before trying to open the browser
sleep 5

# 9. Open Browser (Best effort, uses default Gradio URL)
echo "INFO: Attempting to open $GRADIO_DEFAULT_URL in your default browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
  open "$GRADIO_DEFAULT_URL"
elif command -v xdg-open &> /dev/null; then # Linux standard
  xdg-open "$GRADIO_DEFAULT_URL"
elif command -v cygstart &> /dev/null; then # Cygwin
  cygstart "$GRADIO_DEFAULT_URL"
elif [[ "$OSTYPE" == "msys" ]] && command -v start &> /dev/null; then # Git Bash on Windows might have 'start'
   start "" "$GRADIO_DEFAULT_URL" # Needs empty title argument for start
else
  echo "WARN: Could not automatically open browser. Please navigate to the URL shown above manually."
fi

# Wait for the Gradio process to finish (e.g., user presses Ctrl+C)
# This keeps the script running until the app is stopped.
wait $GRADIO_PID
echo "" # Newline after Ctrl+C output from Gradio
echo "INFO: Gradio application stopped."

echo "--- Script finished ---"
exit 0 