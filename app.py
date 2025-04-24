from fasthtml.common import *
import httpx
from dataclasses import dataclass
from starlette.responses import FileResponse, HTMLResponse
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile
import shutil
from pathlib import Path
import os
import logging

# --- Environment Variables --- Needed for backend URL
BACKEND_SERVICE_URL = os.environ.get("BACKEND_SERVICE_URL", "http://localhost:8001") # Default for local dev

# Configure logging for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [FRONTEND] - %(message)s')

# Configure FastHTML with Monster UI (Pico CSS)
hdrs = (
    Script(src="https://unpkg.com/htmx.org@1.9.10"),
    Script(src="https://unpkg.com/hyperscript.org@0.9.12"),
    Link(rel="stylesheet", href="https://unpkg.com/pico.css@latest"),
)
app, rt = fast_app(hdrs=hdrs)

# --- Constants and Setup --- Frontend specific
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
TEMP_PDF_PATH = UPLOAD_DIR / "uploaded_pdf.pdf"
MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- Helper Functions --- Keep UI helpers

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
    """Sends the PDF to the backend service for processing and displays results."""
    logging.info("Frontend submit endpoint called.")
    should_generate_images = generate_images == "true"
    should_calculate_vacancy = calculate_vacancy == "true" # Convert to boolean
    logging.info(f"Generate images requested: {should_generate_images}")
    logging.info(f"Calculate vacancy requested: {should_calculate_vacancy}") # Log the new flag

    if not TEMP_PDF_PATH.exists():
        logging.error("Error: No valid temporary PDF found at %s.", TEMP_PDF_PATH)
        return Div("Error: No valid PDF file found. Please upload again.", id="results-area")

    try:
        logging.info(f"Sending PDF to backend service at {BACKEND_SERVICE_URL}...")
        backend_process_url = f"{BACKEND_SERVICE_URL}/process_pdf"

        # Read the temporary PDF file content
        pdf_content = TEMP_PDF_PATH.read_bytes()

        # Prepare data payload for the backend request
        data = {
            'generate_images': str(should_generate_images).lower(),
            'calculate_vacancy': str(should_calculate_vacancy).lower()
        }
        # Prepare files payload
        files = {'pdf_file': (TEMP_PDF_PATH.name, pdf_content, 'application/pdf')}

        # Use httpx.AsyncClient for async request
        async with httpx.AsyncClient(timeout=300.0) as client: # Increased timeout for potentially long processing
            response = await client.post(backend_process_url, data=data, files=files)

        # --- Process Backend Response ---
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        backend_results = response.json() # Expecting JSON from backend

        if not backend_results or "results" not in backend_results or not backend_results["results"]:
            msg = backend_results.get("message", "Backend processed the file but returned no unit results.")
            logging.warning(f"Backend returned no results. Message: {msg}")
            # Optionally include run_output_dir info if backend provides it
            run_dir = backend_results.get('run_output_dir')
            if run_dir:
                 msg += f" Backend output directory hint: {run_dir}."
            return Div(msg, id="results-area")

        pipeline_results = backend_results["results"] # List of unit dictionaries
        run_output_dir = backend_results.get("run_output_dir", "unknown_run") # Get the run dir name for downloads
        logging.info(f"Backend returned {len(pipeline_results)} results. Generating UI.")

        # --- Build Results UI ---
        results_html = []
        for unit in pipeline_results:
            # Paths are now RELATIVE paths provided by the backend
            # We need the full backend URL to construct download links
            relative_csv_path = unit.get('csv_path')
            relative_img_paths = unit.get('img_paths', [])
            unit_name = unit.get('unit_name', 'Unknown Unit')
            # Get structured preview data instead of HTML string
            csv_preview_data = unit.get('csv_preview_data') # dict with keys: headers, rows, error

            # --- CSV Download Link ---
            csv_download_url = None
            if relative_csv_path:
                # Construct URL pointing to the backend's download endpoint
                csv_download_url = f"{BACKEND_SERVICE_URL}/download?path={relative_csv_path}"

            # --- CSV Preview Table Generation ---
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
                # Display error message from backend preview generation
                preview_table_component = P(f"Preview Error for {unit_name}: {csv_preview_data['error']}")
            else:
                # Fallback if preview data is missing entirely
                preview_table_component = P(f"Preview data unavailable for {unit_name}.")

            # --- Image Links ---
            image_links_components = []
            if relative_img_paths:
                 image_links_components.append(H4("Generated Images:"))
                 img_list_items = []
                 for img_path in relative_img_paths:
                     # Construct URL pointing to the backend's download endpoint
                     img_download_url = f"{BACKEND_SERVICE_URL}/download?path={img_path}"
                     img_name = Path(img_path).name # Extract filename
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
                     A(
                         "Download Unit CSV",
                         href=csv_download_url,
                         role="button",
                         download=Path(relative_csv_path).name, # Suggest filename
                         cls="outline"
                     )
                 )
            unit_article_content.extend(image_links_components)
            results_html.append(Article(*unit_article_content))

        # Add "Download All" button if results were generated
        if results_html:
             # This button points to the backend's zip download endpoint
             all_download_url = f"{BACKEND_SERVICE_URL}/download_all?run_dir={run_output_dir}"
             results_html.append(
                 A(
                     "Download All Files (.zip)",
                     href=all_download_url,
                     role="button",
                     download=f"{run_output_dir}_all_files.zip", # Suggest filename
                     cls="contrast"
                 )
             )

        return Div(*results_html, id="results-area")

    except httpx.HTTPStatusError as e:
        # Handle HTTP errors from the backend
        error_message = f"Backend service error: {e.response.status_code} - {e.response.text}"
        logging.error(error_message, exc_info=True)
        # Try to parse backend error message if JSON
        try:
            backend_error = e.response.json()
            detail = backend_error.get("detail", e.response.text)
            error_message = f"Backend Error ({e.response.status_code}): {detail}"
        except:
            pass # Keep original text if not JSON
        return Div(error_message, id="results-area")

    except httpx.RequestError as e:
        # Handle network errors connecting to the backend
        error_message = f"Error connecting to backend service: {e}"
        logging.error(error_message, exc_info=True)
        return Div(error_message, id="results-area")

    except Exception as e:
        # Handle unexpected errors during frontend processing or backend communication
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        return Div(f"An unexpected error occurred: {e}", id="results-area")

@rt("/reset")
async def post_reset():
    """Clears the frontend temporary file and resets the UI. Optionally calls backend reset."""
    logging.info("Resetting frontend application state.") # Changed print to logging

    # --- Frontend Cleanup --- 
    # Remove the temp uploaded PDF file if it exists
    if TEMP_PDF_PATH.exists():
        try:
             TEMP_PDF_PATH.unlink()
             logging.info(f"Deleted temporary file: {TEMP_PDF_PATH}")
        except OSError as e:
             logging.warning(f"Could not delete temporary PDF {TEMP_PDF_PATH}: {e}")

    # Remove any other files directly in the frontend's upload dir (optional)
    # for item in UPLOAD_DIR.iterdir():
    #     if item.is_file(): item.unlink()

    # --- Optional: Call Backend Reset Endpoint --- 
    backend_reset_url = f"{BACKEND_SERVICE_URL}/reset_backend"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
             logging.info(f"Sending reset request to backend: {backend_reset_url}")
             response = await client.post(backend_reset_url)
             response.raise_for_status()
             logging.info(f"Backend reset successful: {response.status_code}")
    except httpx.RequestError as e:
         logging.warning(f"Could not connect to backend reset endpoint: {e}")
    except httpx.HTTPStatusError as e:
         logging.warning(f"Backend reset endpoint returned error {e.response.status_code}: {e.response.text}")
    except Exception as e:
         logging.warning(f"An error occurred during backend reset call: {e}")

    # --- Return Initial UI State --- 
    logging.info("Returning initial form state.")
    return get_initial_form_area()


# --- Run the app ---
if __name__ == "__main__":
    serve() 