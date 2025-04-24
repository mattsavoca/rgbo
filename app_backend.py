# --- Backend Service ---
from fasthtml.common import *
import httpx # Potentially needed? Maybe not directly here.
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
from typing import List, Dict, Optional

# Import the actual processing logic
try:
    from pdf_handler import run_pdf_processing
    # Ensure pdf_handler itself can find its dependencies (pdf_pipeline, calculate_vacancy...)
except ImportError as e:
    logging.error(f"Backend FATAL: Failed to import pdf_handler: {e}")
    # Define a dummy function to allow app to potentially start but fail gracefully
    def run_pdf_processing(*args, **kwargs):
        logging.error("run_pdf_processing unavailable due to import error.")
        raise RuntimeError("PDF processing core function failed to load.")

# Configure logging for the backend
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [BACKEND] - %(message)s')

# --- Configuration ---
# Directory to store processed results (CSVs, images, zips)
# This path is relative to where app_backend.py runs (inside the container)
BACKEND_OUTPUT_DIR = Path("backend_data")
BACKEND_OUTPUT_DIR.mkdir(exist_ok=True)

# --- FastHTML App Setup ---
# No special headers needed for a backend API unless providing a UI itself
app, rt = fast_app()

# --- Helper Functions ---

def read_csv_preview(csv_path: Path, max_rows=4) -> Dict:
    """Reads CSV header and first few rows for preview."""
    preview_data = {"headers": [], "rows": [], "error": None}
    if not csv_path.exists():
        preview_data["error"] = f"CSV file not found: {csv_path.name}"
        logging.warning(preview_data["error"])
        return preview_data

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            header_line = next(f, None)
            if not header_line:
                preview_data["error"] = f"CSV file is empty: {csv_path.name}"
                return preview_data

            headers = header_line.strip().split(',')
            preview_data["headers"] = headers
            data_rows = []
            va_index = -1
            try:
                va_index = headers.index("Vacancy Allowance")
            except ValueError:
                pass # Column not present

            for i, line in enumerate(f):
                if i >= max_rows:
                    break
                row = line.strip().split(',')
                if len(row) == len(headers):
                    # Format Vacancy Allowance if index was found
                    if va_index != -1:
                         try:
                             va_float = float(row[va_index])
                             row[va_index] = f"{va_float:.2%}" if va_float != 0 else "0.00%"
                         except (ValueError, TypeError, IndexError):
                             # Keep original string if not a valid float or index issue
                             pass
                    data_rows.append(row)
                else:
                    logging.warning(f"Row/Header length mismatch in {csv_path.name} (row {i+2}): {len(row)} vs {len(headers)}. Skipping row in preview.")
            preview_data["rows"] = data_rows

    except Exception as e:
        logging.error(f"Error reading CSV preview for {csv_path}: {e}")
        preview_data["error"] = f"Could not read preview: {e}"

    return preview_data


# --- API Endpoints ---

@rt("/process_pdf")
async def post_process_pdf(
    pdf_file: UploadFile = Form(...),
    generate_images: str = Form("false"), # Default to false if not provided
    calculate_vacancy: str = Form("false") # Default to false
):
    """Receives PDF, processes it using pdf_handler, returns results as JSON."""
    logging.info("Backend /process_pdf endpoint called.")
    should_generate_images = generate_images.lower() == "true"
    should_calculate_vacancy = calculate_vacancy.lower() == "true"
    logging.info(f"Generate images flag: {should_generate_images}")
    logging.info(f"Calculate vacancy flag: {should_calculate_vacancy}")

    # Save the uploaded file temporarily within the backend's storage
    # Use a simple name for now, consider unique names if handling concurrent requests
    temp_pdf_path = BACKEND_OUTPUT_DIR / (pdf_file.filename or "uploaded_pdf.pdf")
    try:
        file_content = await pdf_file.read()
        if not file_content:
             logging.error("Received empty file upload.")
             return JSONResponse({"error": "Uploaded file is empty."}, status_code=400)
        temp_pdf_path.write_bytes(file_content)
        logging.info(f"Temporarily saved uploaded PDF to {temp_pdf_path}")

    except Exception as e:
        logging.error(f"Failed to save uploaded PDF: {e}", exc_info=True)
        # Use status_code parameter for JSONResponse
        return JSONResponse({"error": f"Failed to save uploaded file: {e}"}, status_code=500)

    # --- Run the actual PDF processing --- 
    processed_results = []
    run_output_dir_relative_path = None
    processing_error = None
    try:
        # run_pdf_processing expects the base output dir (BACKEND_OUTPUT_DIR here)
        # It creates a subdirectory based on the PDF name within that base dir.
        # It returns: (results_list, relative_path_to_run_dir)
        # Example: (results, Path('pdf_file_stem')) where results contain paths
        # relative to BACKEND_OUTPUT_DIR (e.g., 'pdf_file_stem/apt_1.csv')
        pipeline_results, run_output_dir_relative_path = run_pdf_processing(
            temp_pdf_path,
            BACKEND_OUTPUT_DIR,
            generate_images=should_generate_images,
            calculate_vacancy=should_calculate_vacancy
        )

        if run_output_dir_relative_path is None:
            processing_error = "PDF handler failed to initialize or process the PDF."
            logging.error(processing_error)

        elif not pipeline_results:
            logging.warning(f"Pipeline ran for {temp_pdf_path.name} but found no units.")
            # No error, just no results. Frontend will handle message.

        else:
            logging.info(f"PDF handler returned {len(pipeline_results)} results for {temp_pdf_path.name}.")
            # Prepare results for JSON, including CSV preview data
            for unit in pipeline_results:
                unit_result = {
                    'unit_name': unit.get('unit_name', 'Unknown Unit'),
                    'csv_path': str(unit.get('csv_path', '')), # Ensure paths are strings
                    'img_paths': [str(p) for p in unit.get('img_paths', [])], # Ensure paths are strings
                    'csv_preview_data': None
                }
                csv_path_str = unit.get('csv_path')
                if csv_path_str:
                    # Construct full path for reading preview
                    full_csv_path = BACKEND_OUTPUT_DIR / csv_path_str
                    unit_result['csv_preview_data'] = read_csv_preview(full_csv_path)
                processed_results.append(unit_result)

    except Exception as e:
        logging.error(f"Error during PDF processing pipeline: {e}", exc_info=True)
        processing_error = f"An unexpected error occurred during PDF processing: {e}"

    finally:
        # Clean up the temporarily saved PDF file
        if temp_pdf_path.exists():
            try:
                temp_pdf_path.unlink()
            except OSError as e:
                logging.warning(f"Could not delete temporary backend PDF {temp_pdf_path}: {e}")

    # --- Construct and return JSON Response --- 
    if processing_error:
         # Send back a 500 error if processing failed
         return JSONResponse({"error": processing_error}, status_code=500)
    else:
        response_data = {
            "results": processed_results,
            "run_output_dir": str(run_output_dir_relative_path) if run_output_dir_relative_path else None,
            "message": "Processing complete." if processed_results else "Processing complete, but no units found."
        }
        return JSONResponse(response_data)

@rt("/download")
async def get_download_file(path: str):
    """Downloads a single generated file (CSV or image). Path is relative to BACKEND_OUTPUT_DIR."""
    logging.info(f"Backend /download request for path: {path}")
    if not path or '..' in path or '\\' in path: # Basic security checks
        logging.warning(f"Invalid download path characters: {path}")
        return HTMLResponse("Invalid file path", status_code=400)

    # Paths provided by /process_pdf are relative to BACKEND_OUTPUT_DIR
    # e.g., "pdf_file_stem/apt_1.csv" or "pdf_file_stem/pdf_file_stem_images/page_1.png"
    file_path = (BACKEND_OUTPUT_DIR / path).resolve()

    # Security check: Ensure resolved path is still within BACKEND_OUTPUT_DIR
    if not file_path.is_file() or not str(file_path).startswith(str(BACKEND_OUTPUT_DIR.resolve())):
        logging.error(f"Download attempt failed: File not found or outside scope. Requested: {path}, Resolved: {file_path}")
        return HTMLResponse("File not found or access denied", status_code=404)

    filename = file_path.name # Extract filename for download suggestion
    # Basic media type detection
    media_type = 'text/csv' if filename.endswith('.csv') else \
                 ('image/png' if filename.endswith('.png') else \
                  ('image/jpeg' if filename.lower().endswith(('.jpg', '.jpeg')) else \
                   'application/octet-stream'))

    logging.info(f"Serving file: {file_path} as {filename} with media type {media_type}")
    return FileResponse(file_path, filename=filename, media_type=media_type)

@rt("/download_all")
async def get_download_all(run_dir: str):
    """Creates a zip archive of all generated files for a specific run and downloads it."""
    logging.info(f"Backend /download_all request for run_dir: {run_dir}")

    if not run_dir or '..' in run_dir or '\\' in run_dir: # Basic security checks
        logging.warning(f"Invalid run_dir characters: {run_dir}")
        return HTMLResponse("Invalid run directory name", status_code=400)

    # The target directory to zip is relative to BACKEND_OUTPUT_DIR
    target_dir_path = (BACKEND_OUTPUT_DIR / run_dir).resolve()

    # Security check: Ensure target directory exists and is within BACKEND_OUTPUT_DIR
    if not target_dir_path.is_dir() or not str(target_dir_path).startswith(str(BACKEND_OUTPUT_DIR.resolve())):
        logging.error(f"Download all failed: Directory not found or outside scope. Requested: {run_dir}, Resolved: {target_dir_path}")
        return HTMLResponse("Run directory not found or access denied", status_code=404)

    # Create zip file in the base output dir (temporary)
    zip_filename_base = f"{run_dir}_all_files"
    zip_filepath = BACKEND_OUTPUT_DIR / f"{zip_filename_base}.zip"
    files_found = False

    logging.info(f"Attempting to create zip file for {target_dir_path} at {zip_filepath}")
    try:
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            # Iterate only within the specific run directory
            for item in target_dir_path.rglob('*'):
                if item.is_file():
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
        # Ensure filename in response doesn't include the full path
        return FileResponse(zip_filepath, filename=zip_filepath.name, media_type='application/zip', background=BackgroundTask(os.remove, zip_filepath))

    except Exception as e:
        logging.error(f"Error creating zip file for {run_dir}: {e}", exc_info=True)
        if zip_filepath.exists():
            try: os.remove(zip_filepath)
            except OSError as re: logging.error(f"Error removing partial zip: {re}")
        return HTMLResponse("Error creating download package.", status_code=500)

@rt("/reset_backend", methods=["POST"]) # Use POST for actions with side-effects
async def post_reset_backend():
    """Clears the backend processing output directory."""
    logging.info("Backend /reset_backend endpoint called.")
    items_deleted = 0
    items_failed = 0

    if not BACKEND_OUTPUT_DIR.exists():
        logging.info("Backend output directory does not exist, nothing to clear.")
        return JSONResponse({"message": "Output directory already clear."}, status_code=200)

    for item in BACKEND_OUTPUT_DIR.iterdir():
        try:
            if item.is_file():
                item.unlink()
                items_deleted += 1
            elif item.is_dir():
                shutil.rmtree(item)
                items_deleted += 1
        except PermissionError as e:
            items_failed += 1
            logging.warning(f"Could not remove item during reset (likely in use): {item} - {e}")
        except OSError as e:
            items_failed += 1
            logging.warning(f"Could not remove item during reset: {item} - {e}")

    # Recreate the base directory just in case
    BACKEND_OUTPUT_DIR.mkdir(exist_ok=True)

    msg = f"Backend reset attempted. Deleted {items_deleted} items."
    if items_failed > 0:
        msg += f" Failed to delete {items_failed} items (check logs)."
        logging.warning(msg)
        # Still return 200 as reset was attempted, but indicate partial failure
        return JSONResponse({"message": msg, "status": "partial_failure"}, status_code=200)
    else:
        logging.info(msg)
        return JSONResponse({"message": msg, "status": "success"}, status_code=200)

# Add uvicorn execution for running locally / via Docker CMD
if __name__ == "__main__":
    # Use port 8001 for local backend dev, distinct from frontend's 5001
    # The Dockerfile CMD will override this with port 8000 and host 0.0.0.0
    import uvicorn
    uvicorn.run("app_backend:app", host="127.0.0.1", port=8001, reload=True)

# Placeholder for other routes 