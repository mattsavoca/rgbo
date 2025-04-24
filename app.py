from fasthtml.common import *
from dataclasses import dataclass
from starlette.responses import FileResponse
from starlette.background import BackgroundTask # Import BackgroundTask explicitly
import shutil
import zipfile
from pathlib import Path
import os
import logging # Import logging
from pdf_handler import run_pdf_processing # Import the real handler function

# Configure logging for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [APP] - %(message)s')

# Configure FastHTML with Monster UI (Pico CSS)
hdrs = (
    Script(src="https://unpkg.com/htmx.org@1.9.10"),
    Script(src="https://unpkg.com/hyperscript.org@0.9.12"),
    Link(rel="stylesheet", href="https://unpkg.com/pico.css@latest"),
)
app, rt = fast_app(hdrs=hdrs)

# --- Constants and Setup ---
UPLOAD_DIR = Path("temp_folder")
UPLOAD_DIR.mkdir(exist_ok=True) # Create the temp folder if it doesn't exist
TEMP_PDF_PATH = UPLOAD_DIR / "uploaded_pdf.pdf"
MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- Poppler Path (for Windows) ---
# Ensure this relative path is correct from the location where app.py is run
POPPLER_PATH = Path("./poppler-24.08.0/Library/bin")
POPPLER_ENV_VAR = "POPPLER_BIN_PATH" # Environment variable name pdf_to_images will check

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
        "hx_post": "/submit",
        "hx_target": "#results-area",
        "hx_swap": "innerHTML"
    }
    if not enabled:
        btn_attrs["disabled"] = True

    return Div(
        Button("Submit", **btn_attrs),
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
                cls="pico" # Monster UI class might need refinement
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
        # Reset button
        Button("Reset", type="button", hx_post="/reset", hx_target="#main-content", hx_swap="innerHTML", cls="secondary"),
        # Area to display results later
        Div(id="results-area"),
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

# --- Route Handlers ---
@rt("/")
def get():
    """Serves the main page."""
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
    print(f"Checking file: {pdf_file.filename}, size: {pdf_file.size}, type: {pdf_file.content_type}")
    is_valid, message = pdf_check(pdf_file)
    if is_valid:
        # Save the valid file temporarily for the submit step
        try:
            file_bytes = await pdf_file.read()
            TEMP_PDF_PATH.write_bytes(file_bytes)
            print(f"Temporarily saved valid file to {TEMP_PDF_PATH}")
            return get_submit_button(enabled=True, message=message)
        except Exception as e:
            print(f"Error saving temporary file: {e}")
            # Fallback to disabled state if saving fails
            return get_submit_button(enabled=False, message=f"Error saving file: {e}")
    else:
        # If invalid, ensure any previous temp file is gone
        if TEMP_PDF_PATH.exists():
            TEMP_PDF_PATH.unlink()
        return get_submit_button(enabled=False, message=message)

@rt("/submit")
async def post_submit(
    generate_images: str = Form(None),
    calculate_vacancy: str = Form(None) # Read the new checkbox
):
    """Processes the PDF, generates results, and returns the results UI."""
    logging.info("Submit endpoint called.")
    should_generate_images = generate_images == "true"
    should_calculate_vacancy = calculate_vacancy == "true" # Convert to boolean
    logging.info(f"Generate images requested: {should_generate_images}")
    logging.info(f"Calculate vacancy requested: {should_calculate_vacancy}") # Log the new flag

    if not TEMP_PDF_PATH.exists():
        logging.error("Error: No valid PDF found to process at %s.", TEMP_PDF_PATH)
        # Return an error message to the results area
        return Div("Error: No valid PDF file was uploaded or found. Please upload again.", id="results-area")

    # --- Environment Variable Setup for Poppler ---
    original_poppler_path = os.environ.get(POPPLER_ENV_VAR)
    poppler_path_set = False
    if should_generate_images:
        # Check if Poppler path exists before setting env var
        if POPPLER_PATH.exists() and POPPLER_PATH.is_dir():
            poppler_path_str = str(POPPLER_PATH.resolve())
            os.environ[POPPLER_ENV_VAR] = poppler_path_str
            poppler_path_set = True
            logging.info(f"Set environment variable {POPPLER_ENV_VAR}={poppler_path_str}")
        else:
            logging.warning(f"Poppler path specified in app.py does not exist or is not a directory: {POPPLER_PATH}")
            logging.warning("Image generation might fail if Poppler is not in system PATH.")
            # Proceed anyway, assuming it might be in system PATH

    # --- Run Processing --- 
    try:
        # Run the real PDF pipeline via the handler
        logging.info("Calling PDF handler to process %s", TEMP_PDF_PATH)
        # run_pdf_processing returns (results_list, run_output_dir_relative_path)
        pipeline_results, run_dir_rel_path = run_pdf_processing(
            TEMP_PDF_PATH, 
            UPLOAD_DIR, 
            generate_images=should_generate_images,
            calculate_vacancy=should_calculate_vacancy # Pass the flag here
        )

        if run_dir_rel_path is None: # Indicates early failure in handler
             logging.error("PDF handler failed to initialize or find the PDF.")
             return Div("Error: Failed to start PDF processing.", id="results-area")

        if not pipeline_results:
            logging.warning("Pipeline ran but found no units in PDF: %s", TEMP_PDF_PATH)
            # Check if the run directory exists - maybe processing failed mid-way?
            run_dir_abs_path = UPLOAD_DIR / run_dir_rel_path
            if run_dir_abs_path.exists():
                 msg = f"No apartment units found in the PDF. Processed files (if any) are in {run_dir_rel_path}. Please check the PDF content." 
            else:
                 msg = "No apartment units found in the PDF, and no output directory was created. Check processing logs."
            return Div(msg, id="results-area")

        logging.info("PDF handler returned %d results. Generating UI.", len(pipeline_results))
        # Build the results UI
        results_html = []
        for unit in pipeline_results:
            # Paths are relative to UPLOAD_DIR, e.g., Path('pdf_stem/apt_Unit_X.csv')
            relative_csv_path = unit['csv_path'] 
            relative_img_paths = unit.get('img_paths', []) # Get image paths
            unit_name = unit['unit_name']
            full_csv_path = UPLOAD_DIR / relative_csv_path # Create full path for reading

            # Basic CSV preview (first 5 lines)
            preview_table = P(f"CSV: {relative_csv_path.name}") # Default text
            if full_csv_path.exists():
                try:
                    with open(full_csv_path, 'r', encoding='utf-8') as f:
                        header_line = next(f, None) # Read header, default to None if empty
                        if header_line:
                            headers = header_line.strip().split(',')
                            data_rows = []
                            # Read up to 4 data lines for preview
                            for _ in range(4):
                                data_line = next(f, None)
                                if data_line:
                                    row = data_line.strip().split(',')
                                    # Basic validation: ensure same number of columns as header
                                    if len(row) == len(headers):
                                        # Try to format Vacancy Allowance if it exists and is numeric
                                        try:
                                             va_index = headers.index("Vacancy Allowance")
                                             # Attempt to format as percentage, handle errors/strings gracefully
                                             try:
                                                  va_float = float(row[va_index])
                                                  # Format non-zero floats as percentages, leave 0 as 0.0%, handle strings
                                                  if va_float != 0:
                                                       row[va_index] = f"{va_float:.2%}" 
                                                  else:
                                                       row[va_index] = "0.00%" # Explicitly format zero
                                             except (ValueError, TypeError):
                                                  # Keep original string if it's not a valid float (e.g., "Error", "Indeterminable")
                                                  pass 
                                        except ValueError:
                                             pass # Column "Vacancy Allowance" not found in header, ignore formatting
                                        data_rows.append(row)
                                    else:
                                         # Log mismatch and skip row for preview consistency
                                         logging.warning(f"Row/Header length mismatch in {relative_csv_path.name}: {len(row)} vs {len(headers)}. Skipping row in preview.")
                                         # data_rows.append(row + [''] * (len(headers) - len(row))) # Alternative: Pad row
                                else:
                                    break # End of file reached

                            if headers: # Ensure header was actually read
                                preview_table = Table(
                                    Thead(Tr(Th(h) for h in headers)),
                                    Tbody(
                                        (Tr(Td(d) for d in row) for row in data_rows)
                                    )
                                )
                            else: # Should not happen if header_line was not None, but safety check
                                 preview_table = P(f"CSV {relative_csv_path.name} has no header.")
                        else: # File was completely empty
                             preview_table = P(f"CSV {relative_csv_path.name} is empty.")
                except Exception as e:
                    logging.error(f"Error reading CSV preview for {full_csv_path}: {e}")
                    preview_table = P(f"Could not display preview for {relative_csv_path.name}")
            else:
                logging.warning(f"CSV file specified in results not found: {full_csv_path}")
                preview_table = P(f"CSV file not found: {relative_csv_path.name}")

            # --- Image Links --- 
            image_links = []
            if relative_img_paths:
                 image_links.append(H4("Generated Images:"))
                 img_list_items = []
                 for img_path in relative_img_paths:
                    img_download_url = f"/download_unit?path={str(img_path)}"
                    img_list_items.append(
                        Li(A(img_path.name, href=img_download_url, download=img_path.name))
                    )
                 image_links.append(Ul(*img_list_items))
            # --- End Image Links ---

            # Prepare download link (path must be string for URL)
            download_url = f"/download_unit?path={str(relative_csv_path)}"
            results_html.append(
                Article( # Use Article for grouping unit results
                    H3(unit_name),
                    H4("CSV Preview:"),
                    preview_table,
                    A(
                        "Download Unit CSV",
                        href=download_url,
                        role="button", # Pico button styling
                        download=relative_csv_path.name, # Suggest filename to browser
                        cls="outline"
                    ),
                    *image_links # Add the image links section
                )
            )

        # Add "Download All" button if results were generated
        if results_html:
             # This button will trigger download_all which zips UPLOAD_DIR contents
             results_html.append(
                 A(
                     "Download All Files (.zip)",
                     href="/download_all",
                     role="button",
                     download=f"{run_dir_rel_path}_all_files.zip", # Suggest filename based on PDF name
                     cls="contrast" # Different style for main action
                 )
             )

        return Div(*results_html, id="results-area") # Return the list of components

    except Exception as e:
        logging.error(f"Error during PDF pipeline call or results generation: {e}", exc_info=True)
        # Provide feedback to the user
        return Div(f"An unexpected error occurred during processing: {e}. Check application logs.", id="results-area")
    finally:
        # --- Environment Variable Cleanup --- 
        if poppler_path_set:
            if original_poppler_path is None:
                # If it didn't exist before, remove it
                del os.environ[POPPLER_ENV_VAR]
                logging.info(f"Removed environment variable {POPPLER_ENV_VAR}")
            else:
                # Otherwise, restore its original value
                os.environ[POPPLER_ENV_VAR] = original_poppler_path
                logging.info(f"Restored environment variable {POPPLER_ENV_VAR} to original value.")
        elif should_generate_images and original_poppler_path is not None and POPPLER_PATH.exists():
            # If we intended to set it but didn't (e.g., POPPLER_PATH invalid), 
            # ensure we don't accidentally leave a pre-existing value if it wasn't ours.
            # This case is less likely but safe to handle.
            # Re-check if it matches the original path before restoring, though this check might be overkill.
            pass # Or potentially restore original_poppler_path if needed, logic depends on desired env behavior

@rt("/download_unit")
async def get_download_unit(path: str):
    """Downloads a single file (CSV or image). Path is relative to UPLOAD_DIR."""
    if not path or '..' in path: # Basic security check
         logging.warning("Invalid download path requested: %s", path)
         return HTML("Invalid path", status_code=400)

    file_path = (UPLOAD_DIR / path).resolve() # Resolve to absolute path
    
    # Security check: Ensure resolved path is still within UPLOAD_DIR
    if not file_path.is_file() or not str(file_path).startswith(str(UPLOAD_DIR.resolve())):
        logging.error("Download attempt failed: File not found or outside upload dir: %s (resolved: %s)", path, file_path)
        return HTML("File not found or access denied", status_code=404)
        
    filename = Path(path).name # Extract filename for download suggestion
    media_type = 'text/csv' if filename.endswith('.csv') else ('image/png' if filename.endswith('.png') else 'application/octet-stream') # Basic media type detection
    logging.info(f"Downloading unit file: {file_path} as {filename} with media type {media_type}")
    return FileResponse(file_path, filename=filename, media_type=media_type)

@rt("/download_all")
async def get_download_all():
    """Creates a zip archive of all generated files in UPLOAD_DIR and downloads it."""
    zip_filename_base = "all_files"
    # Try to find the run directory name to make the zip filename more specific
    # This assumes there's only one run's output in UPLOAD_DIR due to reset logic
    run_dirs = [d for d in UPLOAD_DIR.iterdir() if d.is_dir()] 
    if len(run_dirs) == 1:
        zip_filename_base = f"{run_dirs[0].name}_all_files"
    
    zip_filename = UPLOAD_DIR / f"{zip_filename_base}.zip"
    files_found = False

    logging.info(f"Attempting to create zip file: {zip_filename}")
    try:
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            # Iterate through the UPLOAD_DIR recursively
            for item in UPLOAD_DIR.rglob('*'):
                if item.is_file() and item.suffix.lower() in ['.csv', '.png', '.jpg', '.jpeg']:
                    # Calculate arcname relative to UPLOAD_DIR to preserve structure
                    arcname = item.relative_to(UPLOAD_DIR)
                    zipf.write(item, arcname=arcname)
                    logging.debug(f"Adding to zip: {item} as {arcname}")
                    files_found = True

        if not files_found:
             logging.warning("No compatible files (.csv, .png, .jpg, .jpeg) found in %s to zip.", UPLOAD_DIR)
             # Clean up empty zip file
             if zip_filename.exists(): os.remove(zip_filename) 
             return HTML("No files found to download.", status_code=404)

        logging.info(f"Zip file created successfully: {zip_filename}")
        # Return the zip file and delete it afterwards using BackgroundTask
        return FileResponse(zip_filename, filename=zip_filename.name, media_type='application/zip', background=BackgroundTask(os.remove, zip_filename))

    except Exception as e:
        logging.error(f"Error creating zip file: {e}", exc_info=True)
        # Clean up partial zip if it exists
        if zip_filename.exists():
            try:
                os.remove(zip_filename)
            except OSError as remove_err:
                 logging.error(f"Error removing partial zip file {zip_filename}: {remove_err}")
        return HTML("Error creating download package.", status_code=500)


@rt("/reset")
async def post_reset():
    """Clears the temporary folder and resets the UI."""
    print("Resetting application state.")
    # Remove the temp directory contents
    if TEMP_PDF_PATH.exists():
        try:
             TEMP_PDF_PATH.unlink()
        except OSError as e:
             # Log if even the single temp PDF fails deletion, but continue
             logging.warning(f"Could not delete temporary PDF {TEMP_PDF_PATH}: {e}")

    for item in UPLOAD_DIR.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except PermissionError as e:
            # Specifically catch permission errors (like file-in-use) during rmtree
            logging.warning(f"Could not remove directory {item} during reset (likely in use): {e}")
        except OSError as e:
            # Catch other potential OS errors during deletion
            logging.warning(f"Could not remove item {item} during reset: {e}")

    # Recreate dir just in case it got deleted somehow (though we mostly delete contents)
    UPLOAD_DIR.mkdir(exist_ok=True)

    logging.info("Temporary file cleanup attempted.") # Changed print to logging.info
    # Return the initial form state HTML to replace #main-content
    return get_initial_form_area()


# --- Run the app ---
if __name__ == "__main__":
    serve() 