# --- main.py ---
from fasthtml.common import *
import httpx # Keep for potential external calls if needed in future, but not for internal processing
from dataclasses import dataclass
from starlette.responses import FileResponse, JSONResponse, HTMLResponse # Need JSONResponse
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile
import shutil
import zipfile
from pathlib import Path
import os
import logging
import re
import polars as pl # Needed for preview generation
from typing import List, Dict, Optional, Tuple

# --- Import Core Processing Logic ---
try:
    from pdf_handler import run_pdf_processing
    # Ensure pdf_handler itself can find its dependencies (pdf_pipeline, calculate_vacancy...)
except ImportError as e:
    logging.error(f"App FATAL: Failed to import pdf_handler: {e}")
    # Define a dummy function to allow app to potentially start but fail gracefully
    def run_pdf_processing(*args, **kwargs):
        logging.error("run_pdf_processing unavailable due to import error.")
        raise RuntimeError("PDF processing core function failed to load.")

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [APP] - %(message)s')

# --- Configure FastHTML ---
hdrs = (
    Script(src="https://unpkg.com/htmx.org@1.9.10"),
    Script(src="https://unpkg.com/hyperscript.org@0.9.12"),
    Link(rel="stylesheet", href="https://unpkg.com/pico.css@latest"),
    # Add CSS for the HTMX spinner indicator
    Style("""
        .htmx-indicator {
            display: none; /* Hidden by default */
            width: 1.5em;  /* Adjust size as needed */
            height: 1.5em;
            border: 2px solid currentColor;
            border-right-color: transparent;
            border-radius: 50%;
            animation: spin .75s linear infinite;
            vertical-align: middle; /* Align nicely with button text */
            margin-left: 0.5em; /* Space it from the button */
        }
        /* Show the indicator when HTMX adds the htmx-request class to it */
        .htmx-request.htmx-indicator {
            display: inline-block;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    """)
)
# Note: No explicit default backend URL needed now
app, rt = fast_app(hdrs=hdrs)

# --- Constants and Setup ---
# Vercel (and other serverless environments) typically only allow writing to /tmp
TMP_ROOT = Path("/tmp")
# Directory for temporary uploads within the /tmp directory
UPLOAD_DIR = TMP_ROOT / "temp_uploads"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True) # Added parents=True for robustness
# Directory for processed output within the /tmp directory (ephemeral)
OUTPUT_DIR = TMP_ROOT / "processed_data"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True) # Added parents=True for robustness

TEMP_PDF_PATH = UPLOAD_DIR / "uploaded_pdf.pdf"
MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- Helper Functions ---

def is_pdf(file: UploadFile) -> bool:
    return file.content_type == 'application/pdf'

def is_size_ok(file: UploadFile) -> bool:
    return file.size is not None and file.size <= MAX_FILE_SIZE_BYTES

def pdf_check(file: UploadFile | None) -> tuple[bool, str]:
    """Checks if the file is a valid PDF under the size limit."""
    if not file:
        return False, "No file selected."
    if not is_pdf(file):
        return False, "File must be a PDF."
    if not is_size_ok(file):
        return False, f"File size cannot exceed {MAX_FILE_SIZE_MB} MB."
    return True, "File is valid."

def get_submit_button(enabled: bool, message: str | None = None) -> Div:
    """Returns the submit button div, enabled or disabled, with an optional message."""
    btn_attrs = {
        "type": "submit",
        "hx_post": "/submit", # Calls the submit route in *this* app
        "hx_target": "#results-area",
        "hx_indicator": "#spinner", # Added an indicator
        "hx_swap": "innerHTML"
    }
    if not enabled:
        btn_attrs["disabled"] = True

    return Div(
        Button("Submit", **btn_attrs),
        # Simple loading indicator - now styled via CSS in headers
        # It's hidden by default and shown when htmx-request class is added
        Span(id="spinner", cls="htmx-indicator"), # No default text needed
        P(message, cls="warning") if message and not enabled else '',
        P(message, cls="success") if message and enabled else '',
        id="submit-button-area" # Ensure ID is present for swapping
    )

def get_initial_form_area() -> Form:
    """Returns the initial form state."""
    return Form(
        Div( # Group file input and checkbox
            Label("Select PDF File:", **{'for': 'pdf-input'}),
            Input(
                type="file",
                name="pdf_file",
                id="pdf-input", # Added ID for targeting
                accept=".pdf", # Hint to browser
                cls="pico"
            ),
            Label(
                Input(type="checkbox", id="gen-images-checkbox", name="generate_images", value="true"),
                "Generate page images?"
            ),
            Label(
                Input(type="checkbox", id="calc-vacancy-checkbox", name="calculate_vacancy", value="true"),
                "Calculate Vacancy Allowance? (Alpha)"
            ),
            cls="grid" # Basic layout
        ),
        # Placeholder for the submit button, initially disabled
        get_submit_button(enabled=False, message="Select a PDF file to upload."),
        # Div for simulated log output
        Div(id="log-output", style="margin-top: 1em; border: 1px solid #ccc; padding: 0.5em; min-height: 50px; font-family: monospace; white-space: pre-wrap; overflow-y: auto; background-color: #f8f8f8;"),
        # Reset button
        Button("Reset", type="button", hx_post="/reset", hx_target="#main-content", hx_swap="innerHTML", cls="secondary"),
        # Area to display results later
        Div(id="results-area"),
        # Script for dynamic progress feedback
        Script("""
            let timerInterval = null;
            let startTime = null;
            const form = document.getElementById('upload-form');
            const logOutput = document.getElementById('log-output');

            if (form && logOutput) {
                form.addEventListener('htmx:beforeRequest', function(evt) {
                    // Check if the request is going to /submit
                    if (evt.detail.requestConfig.path === '/submit') {
                        logOutput.textContent = 'Processing started...';
                        startTime = Date.now();
                        // Clear previous interval if any (safety check)
                        if (timerInterval) clearInterval(timerInterval);

                        timerInterval = setInterval(function() {
                            const elapsedSeconds = Math.round((Date.now() - startTime) / 1000);
                            logOutput.textContent = `Processing... Elapsed time: ${elapsedSeconds} seconds.`;
                        }, 1000); // Update every second
                    }
                });

                form.addEventListener('htmx:afterRequest', function(evt) {
                    // Stop timer only if it was started for the /submit request
                    if (evt.detail.requestConfig.path === '/submit') {
                         if (timerInterval) {
                            clearInterval(timerInterval);
                            timerInterval = null;
                         }
                         // Optionally clear or update log message on completion/error
                         // logOutput.textContent = 'Processing finished.'; // Or handled by server response swap
                    }
                });

                form.addEventListener('htmx:responseError', function(evt) {
                    // Also stop timer on error
                    if (evt.detail.requestConfig.path === '/submit') {
                         if (timerInterval) {
                            clearInterval(timerInterval);
                            timerInterval = null;
                         }
                         logOutput.textContent = 'An error occurred during processing.';
                    }
                });
            } else {
                console.error('Could not find form or log output element for timer script.');
            }
        """),
        # Required for file uploads
        enctype="multipart/form-data",
        # Target the whole content area for reset
        id="upload-form",
        # Added HTMX attributes for file check here
        hx_post="/check_pdf",
        hx_target="#submit-button-area",
        hx_swap="outerHTML",
        hx_trigger="change from:#pdf-input" # Trigger check on file input change
    )

def read_csv_preview(csv_path: Path, max_rows=4) -> Dict:
    """Reads CSV header and first few rows for preview (uses OUTPUT_DIR)."""
    preview_data = {"headers": [], "rows": [], "error": None}
    full_csv_path = OUTPUT_DIR / csv_path # Path is relative to OUTPUT_DIR

    if not full_csv_path.exists():
        preview_data["error"] = f"CSV file not found: {full_csv_path.name}"
        logging.warning(preview_data["error"] + f" (Path: {full_csv_path})")
        return preview_data

    try:
        # Use Polars for consistency and robustness
        df_preview = pl.read_csv(full_csv_path, n_rows=max_rows, try_parse_dates=False) # Avoid parsing dates for preview
        preview_data["headers"] = df_preview.columns

        # Format rows, especially the Vacancy Allowance column
        data_rows = []
        va_col_name = "Vacancy Allowance"
        for row_tuple in df_preview.iter_rows():
            row_list = list(row_tuple)
            if va_col_name in preview_data["headers"]:
                 va_index = preview_data["headers"].index(va_col_name)
                 try:
                     va_val = row_list[va_index]
                     if isinstance(va_val, (int, float)):
                         row_list[va_index] = f"{va_val:.2%}"
                     # Keep strings as they are (might be errors like "Indeterminable...")
                 except (ValueError, TypeError, IndexError):
                     pass # Keep original
            data_rows.append([str(item) if item is not None else "" for item in row_list]) # Convert all to strings for display
        preview_data["rows"] = data_rows

        if not preview_data["headers"]:
             preview_data["error"] = f"CSV file appears empty or headerless: {full_csv_path.name}"

    except Exception as e:
        logging.error(f"Error reading CSV preview for {full_csv_path}: {e}", exc_info=True)
        preview_data["error"] = f"Could not read preview: {e}"

    return preview_data

# --- Route Handlers ---
@rt("/")
def get_root():
    """Serves the main page."""
    logging.info("Serving root page.")
    return Titled(
        "DHCR - RGBO Parser",
        Main(
            Div(
                get_initial_form_area(),
                id="main-content" # Target for reset
            ),
            cls="container" # Pico class for centering/padding
        )
    )

@rt("/check_pdf")
async def post_check_pdf(pdf_file: UploadFile = Form(...)):
    """Checks the uploaded file and returns an enabled/disabled submit button."""
    logging.info(f"Checking file: {pdf_file.filename}, size: {pdf_file.size}, type: {pdf_file.content_type}")
    is_valid, message = pdf_check(pdf_file)
    if is_valid:
        # Save the valid file temporarily for the submit step
        try:
            # Ensure the upload dir exists (might be cleared on reset)
            UPLOAD_DIR.mkdir(exist_ok=True)
            file_bytes = await pdf_file.read()
            TEMP_PDF_PATH.write_bytes(file_bytes)
            logging.info(f"Temporarily saved valid file to {TEMP_PDF_PATH}")
            return get_submit_button(enabled=True, message=message)
        except Exception as e:
            logging.error(f"Error saving temporary file: {e}", exc_info=True)
            return get_submit_button(enabled=False, message=f"Error saving file: {e}")
    else:
        # If invalid, ensure any previous temp file is gone
        if TEMP_PDF_PATH.exists():
            try:
                TEMP_PDF_PATH.unlink()
            except OSError as e:
                logging.warning(f"Could not delete invalid temp PDF {TEMP_PDF_PATH}: {e}")
        return get_submit_button(enabled=False, message=message)

@rt("/submit")
async def post_submit(
    generate_images: str = Form(None),
    calculate_vacancy: str = Form(None)
):
    """Processes the uploaded PDF directly and displays results."""
    logging.info("Submit endpoint called for processing.")
    should_generate_images = generate_images == "true"
    should_calculate_vacancy = calculate_vacancy == "true"
    logging.info(f"Generate images requested: {should_generate_images}")
    logging.info(f"Calculate vacancy requested: {should_calculate_vacancy}")

    if not TEMP_PDF_PATH.exists():
        logging.error("Submit error: No valid temporary PDF found at %s.", TEMP_PDF_PATH)
        return Div("Error: No valid PDF file found. Please upload again.", id="results-area")

    # --- Direct Processing ---
    pipeline_results = []
    run_output_dir_relative_path = None
    processing_error = None
    try:
        # Call pdf_handler function directly
        # It will create a subdir within OUTPUT_DIR
        logging.info(f"Starting direct PDF processing for {TEMP_PDF_PATH.name}...")
        pipeline_results, run_output_dir_relative_path = run_pdf_processing(
            TEMP_PDF_PATH,
            OUTPUT_DIR, # Base output directory within the container
            generate_images=should_generate_images,
            calculate_vacancy=should_calculate_vacancy
        )

        if run_output_dir_relative_path is None:
            # This case likely means run_pdf_processing hit an early error or didn't return properly
             processing_error = "PDF handler failed to initialize or complete processing."
             logging.error(processing_error + f" (Input: {TEMP_PDF_PATH.name})")
             # Check logs from pdf_handler/pdf_pipeline for specific errors

        elif not pipeline_results:
            # Processing completed but found no units
            logging.warning(f"Processing complete for {TEMP_PDF_PATH.name}, but no units found.")
            # run_output_dir_relative_path should still be valid here if processing ran

        else:
            # Processing successful, results found
            logging.info(f"Direct processing successful, found {len(pipeline_results)} results for {TEMP_PDF_PATH.name}.")

    except Exception as e:
        logging.error(f"Error during direct PDF processing call: {e}", exc_info=True)
        processing_error = f"An unexpected error occurred during processing: {e}"
        # Attempt to get the run directory name if possible for potential partial cleanup info
        run_output_dir_relative_path = Path(TEMP_PDF_PATH.stem) if not run_output_dir_relative_path else run_output_dir_relative_path


    # --- Build Results UI (or Error Message) ---
    if processing_error:
        # Display error to user
        # Optionally add a button to try reset if helpful
        return Div(P(f"Error processing PDF: {processing_error}"),
                   Button("Reset", type="button", hx_post="/reset", hx_target="#main-content", hx_swap="innerHTML", cls="secondary"),
                   id="results-area")

    if not pipeline_results:
        # Display no results message
        return Div(P(f"Processing complete, but no unit data was extracted from {TEMP_PDF_PATH.name}."),
                   # Provide download all link even if empty, leads to empty zip
                   A( "Download All Files (.zip)",
                       href=f"/download_all?run_dir={run_output_dir_relative_path}",
                       role="button", download=f"{run_output_dir_relative_path}_all_files.zip", cls="contrast secondary",
                       **{"aria-disabled": "true"} # Indicate it might be empty
                   ) if run_output_dir_relative_path else "",
                   id="results-area")

    # --- Build Results UI for successful processing ---
    results_html = []
    run_dir_str = str(run_output_dir_relative_path) if run_output_dir_relative_path else "unknown_run"

    for unit in pipeline_results:
        # Paths are RELATIVE paths provided by pdf_handler (relative to OUTPUT_DIR)
        relative_csv_path = unit.get('csv_path') # e.g., Path('pdf_stem/apt_1.csv')
        relative_img_paths = unit.get('img_paths', []) # e.g., [Path('pdf_stem/images/page_1.png')]
        unit_name = unit.get('unit_name', 'Unknown Unit')

        # --- CSV Download Link ---
        csv_download_url = None
        csv_filename_suggestion = "download.csv"
        if relative_csv_path:
            # Download link points to *this* app's /download route
            # Use as_posix() for URL compatibility
            csv_download_url = f"/download?path={relative_csv_path.as_posix()}"
            csv_filename_suggestion = Path(relative_csv_path).name

        # --- CSV Preview Table Generation ---
        csv_preview_data = None
        if relative_csv_path:
             # Call local helper function
             csv_preview_data = read_csv_preview(relative_csv_path)

        preview_table_component = None
        if csv_preview_data and not csv_preview_data.get("error"):
            headers = csv_preview_data.get("headers", [])
            data_rows = csv_preview_data.get("rows", [])
            if headers and data_rows:
                preview_table_component = Table(
                    Thead(Tr(*(Th(h) for h in headers))),
                    Tbody(*(Tr(*(Td(d) for d in row)) for row in data_rows))
                )
            elif headers:
                 preview_table_component = P(f"Preview for {unit_name}: Header found, but no data rows.")
            else:
                 preview_table_component = P(f"Preview for {unit_name}: No header/data found in preview data.")
        elif csv_preview_data and csv_preview_data.get("error"):
            preview_table_component = P(f"Preview Error for {unit_name}: {csv_preview_data['error']}")
        else:
            preview_table_component = P(f"Preview data unavailable for {unit_name}.")

        # --- Image Links ---
        image_links_components = []
        if relative_img_paths:
             image_links_components.append(H4("Generated Images:"))
             img_list_items = []
             for img_path in relative_img_paths:
                 # Download link points to *this* app's /download route
                 # Use as_posix() for URL compatibility
                 img_download_url = f"/download?path={img_path.as_posix()}"
                 img_name = Path(img_path).name
                 img_list_items.append(
                     Li(A(img_name, href=img_download_url, download=img_name))
                 )
             image_links_components.append(Ul(*img_list_items))

        # --- Assemble Unit Article ---
        unit_article_content = [
             H3(unit_name),
             H4("CSV Preview:"),
             preview_table_component,
        ]
        if csv_download_url:
             unit_article_content.append(
                 A( "Download Unit CSV", href=csv_download_url, role="button",
                    download=csv_filename_suggestion, cls="outline" )
             )
        unit_article_content.extend(image_links_components)
        results_html.append(Article(*unit_article_content))

    # --- Add "Download All" button ---
    if results_html and run_output_dir_relative_path:
        all_download_url = f"/download_all?run_dir={run_dir_str}"
        results_html.append(
            A("Download All Files (.zip)", href=all_download_url, role="button",
              download=f"{run_dir_str}_all_files.zip", cls="contrast")
        )

    return Div(*results_html, id="results-area")


@rt("/download")
async def get_download_file(path: str):
    """Downloads a single generated file (CSV or image). Path is relative to OUTPUT_DIR."""
    logging.info(f"Download request for path: {path}")
    if not path or '..' in path or '\\' in path: # Basic security checks
        logging.warning(f"Invalid download path characters: {path}")
        return HTMLResponse("Invalid file path", status_code=400)

    # Paths provided are relative to OUTPUT_DIR
    # e.g., "pdf_file_stem/apt_1.csv" or "pdf_file_stem/pdf_file_stem_images/page_1.png"
    try:
        # Construct the absolute path within the container's filesystem
        file_path = (OUTPUT_DIR / path).resolve()

        # Security check: Ensure resolved path is still within OUTPUT_DIR
        if not file_path.is_file() or not str(file_path).startswith(str(OUTPUT_DIR.resolve())):
            logging.error(f"Download attempt failed: File not found or outside scope. Requested: {path}, Resolved: {file_path}")
            return HTMLResponse("File not found or access denied", status_code=404)

        filename = file_path.name # Extract filename for download suggestion
        media_type = 'text/csv' if filename.endswith('.csv') else \
                     ('image/png' if filename.endswith('.png') else \
                      ('image/jpeg' if filename.lower().endswith(('.jpg', '.jpeg')) else \
                       'application/octet-stream'))

        logging.info(f"Serving file: {file_path} as {filename} with media type {media_type}")
        # No background task needed for single file download cleanup as the whole dir is ephemeral
        return FileResponse(file_path, filename=filename, media_type=media_type)
    except Exception as e:
         logging.error(f"Error serving download for path {path}: {e}", exc_info=True)
         return HTMLResponse("Error serving file.", status_code=500)


@rt("/download_all")
async def get_download_all(run_dir: str):
    """Creates a zip archive of all generated files for a specific run and downloads it."""
    logging.info(f"Download_all request for run_dir: {run_dir}")

    if not run_dir or '..' in run_dir or '\\' in run_dir: # Basic security checks
        logging.warning(f"Invalid run_dir characters: {run_dir}")
        return HTMLResponse("Invalid run directory name", status_code=400)

    # The target directory to zip is relative to OUTPUT_DIR
    target_dir_path = (OUTPUT_DIR / run_dir).resolve()

    # Security check: Ensure target directory exists and is within OUTPUT_DIR
    if not target_dir_path.is_dir() or not str(target_dir_path).startswith(str(OUTPUT_DIR.resolve())):
        logging.error(f"Download all failed: Directory not found or outside scope. Requested: {run_dir}, Resolved: {target_dir_path}")
        return HTMLResponse("Run directory not found or access denied", status_code=404)

    # Create zip file in the base output dir (temporary, but still ephemeral)
    zip_filename_base = f"{run_dir}_all_files"
    # Place the temporary zip file inside the OUTPUT_DIR as well
    zip_filepath = OUTPUT_DIR / f"{zip_filename_base}.zip"
    files_found = False

    logging.info(f"Attempting to create zip file for {target_dir_path} at {zip_filepath}")
    try:
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Iterate only within the specific run directory
            for item in target_dir_path.rglob('*'):
                if item.is_file() and item != zip_filepath: # Don't zip the zip itself
                    # arcname should be relative to the target_dir to maintain structure inside zip
                    arcname = item.relative_to(target_dir_path)
                    zipf.write(item, arcname=arcname)
                    logging.debug(f"Adding to zip: {item} as {arcname}")
                    files_found = True

        if not files_found:
            logging.warning(f"No files found in {target_dir_path} to zip.")
            if zip_filepath.exists(): os.remove(zip_filepath) # Clean up empty zip
            return HTMLResponse("No files found in run directory to download.", status_code=404)

        logging.info(f"Zip file created successfully: {zip_filepath}")
        # Return the zip file and delete it afterwards using BackgroundTask
        # This ensures the zip is removed even if the download is interrupted,
        # reducing clutter in the ephemeral storage before the instance disappears.
        return FileResponse(zip_filepath, filename=zip_filepath.name, media_type='application/zip', background=BackgroundTask(os.remove, str(zip_filepath)))

    except Exception as e:
        logging.error(f"Error creating zip file for {run_dir}: {e}", exc_info=True)
        if zip_filepath.exists():
            try: os.remove(zip_filepath)
            except OSError as re: logging.error(f"Error removing partial zip: {re}")
        return HTMLResponse("Error creating download package.", status_code=500)


@rt("/reset")
async def post_reset():
    """Clears temporary upload and processed output directories."""
    logging.info("Resetting application state (clearing temp directories).")

    # --- Frontend Temp Upload Cleanup ---
    if TEMP_PDF_PATH.exists():
        try:
             TEMP_PDF_PATH.unlink()
             logging.info(f"Deleted temporary upload file: {TEMP_PDF_PATH}")
        except OSError as e:
             logging.warning(f"Could not delete temporary PDF {TEMP_PDF_PATH}: {e}")
    # Clean the rest of the upload dir
    for item in UPLOAD_DIR.iterdir():
        if item.is_file() and item != TEMP_PDF_PATH: # Avoid trying to delete again
            try: item.unlink()
            except OSError as e: logging.warning(f"Could not delete item in {UPLOAD_DIR}: {item} - {e}")
        elif item.is_dir():
            try: shutil.rmtree(item)
            except OSError as e: logging.warning(f"Could not delete subdir in {UPLOAD_DIR}: {item} - {e}")

    # --- Ephemeral Processed Data Cleanup ---
    # This is technically optional as Cloud Run handles cleanup, but good for immediate reset feel
    items_deleted = 0
    items_failed = 0
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                    items_deleted += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    items_deleted += 1
            except PermissionError as e: # Might occur if file is still being served
                 items_failed+=1
                 logging.warning(f"Could not remove item during reset (likely in use): {item} - {e}")
            except OSError as e:
                 items_failed+=1
                 logging.warning(f"Could not remove item during reset: {item} - {e}")
        logging.info(f"Cleared {items_deleted} items from {OUTPUT_DIR}. Failed to clear {items_failed}.")
    else:
        logging.info(f"{OUTPUT_DIR} does not exist, nothing to clear.")

    # Recreate dirs
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Return Initial UI State ---
    logging.info("Returning initial form state.")
    return get_initial_form_area()


# --- Run the app ---
if __name__ == "__main__":
    # Cloud Run expects the app to listen on port 8080 by default
    # The Dockerfile CMD handles this correctly. `serve()` is for local dev convenience.
    # Use environment variable PORT if set (like Cloud Run does), otherwise default to 5001 for local dev
    port = int(os.environ.get("PORT", 5001))
    # Use 0.0.0.0 to be accessible within Docker/Cloud Run
    serve(host="0.0.0.0", port=port)