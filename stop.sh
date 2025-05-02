#!/bin/bash

# Script to attempt stopping the DHCR Parser Gradio app process.
# NOTE: The primary way to stop the app (when run via run.sh)
# is now to press CTRL+C in the terminal where run.sh is executing.
# This script is a fallback in case the process is detached or CTRL+C failed.

SCRIPT_NAME="gradio_app.py"

echo "--- DHCR Parser Stopper ---"

# Find Process IDs (PIDs) running the Gradio app
# Using pgrep -f to match the full command line argument string
# This is generally more reliable than just matching 'python'
PIDS=$(pgrep -f "python.*${SCRIPT_NAME}")

if [ -z "$PIDS" ]; then
    echo "INFO: No running process found matching 'python.*$SCRIPT_NAME'."
    echo "The application might already be stopped, or it wasn't started."
else
    echo "INFO: Found running process(es) for $SCRIPT_NAME with PID(s): $PIDS"
    echo "Attempting to stop..."

    # Loop through each PID found and try to kill it
    for pid in $PIDS; do
        # Check if the PID still exists before trying to kill
        if ps -p $pid > /dev/null; then
            echo " - Stopping PID: $pid"
            kill $pid # Send TERM signal (graceful shutdown)
            # Wait briefly and check if it stopped
            sleep 1
            if ps -p $pid > /dev/null; then
                echo "   WARN: Process $pid did not stop gracefully. Sending KILL signal..."
                kill -9 $pid # Force kill
                sleep 1 # Give OS a moment
                if ps -p $pid > /dev/null; then
                     echo "   ERROR: Failed to stop PID $pid even with KILL signal."
                else
                     echo "   INFO: Process $pid stopped via KILL signal."
                fi
            else
                echo "   INFO: Process $pid stopped successfully."
            fi
        else
             echo " - PID $pid seems to have already finished."
        fi
    done
    echo "INFO: Stop attempt finished."
fi

echo "--- Stopper finished ---"
exit 0 