import gradio as gr
import tempfile
import zipfile
from pathlib import Path
import shutil
import logging
import os
from typing import Optional, Tuple, Any, List, Dict
import polars as pl # Use polars for reading CSVs

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [GRADIO_APP] - %(message)s')

# Import the calculator tab creator FIRST (after logging config)
try:
    from vacancy_calculator_tab import create_calculator_tab
    # Use logging directly here as it's configured just above
    logging.info("Successfully imported UI builder from vacancy_calculator_tab.")
    calculator_tab_available = True
except ImportError as e:
    logging.error(f"Failed to import vacancy_calculator_tab: {e}. Calculator tab will be disabled.", exc_info=True)
    calculator_tab_available = False
    # Define a dummy function if import fails
    def create_calculator_tab():
        with gr.Blocks() as calculator_tab_error:
            gr.Markdown("## Vacancy Allowance Calculator")
            gr.Markdown("**Error:** Failed to load the calculator tab module. Please check the application logs.")
        return calculator_tab_error

# --- Import Core Logic ---
# Assuming pdf_handler.py is in the same directory
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("Attempting to load environment variables from .env file.")
    from pdf_handler import run_pdf_processing
    logging.info("Successfully imported run_pdf_processing from pdf_handler.")
except ImportError as e:
    logging.error(f"Fatal Error: Failed to import core processing logic: {e}", exc_info=True)
    def run_pdf_processing(*args, **kwargs):
        logging.error("run_pdf_processing function is unavailable due to import error.")
        raise RuntimeError("Core PDF processing logic failed to load. Cannot continue.") from e
except Exception as e:
     logging.error(f"An unexpected error occurred during import or env loading: {e}", exc_info=True)
     def run_pdf_processing(*args, **kwargs):
        logging.error("run_pdf_processing function is unavailable due to an unexpected setup error.")
        raise RuntimeError("Core PDF processing logic failed to load. Cannot continue.") from e

# --- Gradio Processing Function ---

# Update return signature: Add preview_cache_dir_state output
def process_dhcr_pdf(pdf_file_obj, generate_images: bool, calculate_vacancy: bool) -> Tuple[
    Optional[str], str, Optional[str], # zip_output, status_output, zip_path_state
    Dict, Dict, Dict, Dict, # df_preview update, df_group update, unit_selector update, unit_data_map_state update
    Optional[str] # preview_cache_dir_state update
]: # Now returns 8 items
    """
    Main function called by Gradio interface. Processes PDF, copies preview CSVs to a
    persistent cache, zips results, prepares data for the first unit's DataFrame,
    populates unit selector dropdown, and stores a map of unit names to FULL CSV paths
    in the persistent cache.

    Returns:
        A tuple containing updates for various UI components.
    """
    # Initialize return values
    first_unit_df_data = gr.update(value=None)
    df_group_update = gr.update(visible=False)
    unit_selector_update = gr.update(choices=[], value=None, visible=False)
    unit_data_map: Dict[str, Path] = {} # Map unit name to its FULL CSV path IN CACHE
    final_zip_path: Optional[str] = None
    persistent_zip_path_state: Optional[str] = None
    preview_cache_dir: Optional[Path] = None # Path to the persistent preview dir
    preview_cache_dir_state_update: Optional[str] = None

    if pdf_file_obj is None:
        return (None, "Error: No PDF file provided.", None,
                first_unit_df_data, df_group_update, unit_selector_update, unit_data_map,
                preview_cache_dir_state_update)

    uploaded_pdf_path = Path(pdf_file_obj.name)
    logging.info(f"Received file: {uploaded_pdf_path.name}, Generate Images: {generate_images}, Calculate Vacancy: {calculate_vacancy}")

    # --- Create Persistent Preview Cache Directory FIRST ---
    try:
        preview_cache_dir = Path(tempfile.mkdtemp(prefix="gradio_preview_"))
        preview_cache_dir_state_update = str(preview_cache_dir) # Store path for state return
        logging.info(f"Created persistent preview cache directory: {preview_cache_dir}")
    except Exception as e:
        logging.error(f"Failed to create persistent preview cache directory: {e}", exc_info=True)
        status_message = f"Fatal Error: Could not create preview cache directory: {e}"
        # Return error state (8 items)
        return (None, status_message, None,
                gr.update(value=None), gr.update(visible=False),
                gr.update(choices=[], value=None, visible=False), {},
                None)

    try:
        # --- Process PDF in its own temporary directory ---
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            temp_input_pdf = temp_dir / uploaded_pdf_path.name
            # This is where run_pdf_processing will create output/<pdf_stem>
            temp_output_base = temp_dir / "output"
            temp_output_base.mkdir()
            shutil.copy(uploaded_pdf_path, temp_input_pdf)
            logging.info(f"Copied uploaded PDF to temporary input: {temp_input_pdf}")

            status_message = f"Processing {temp_input_pdf.name}...\n"
            status_message += f"Generate Images: {generate_images}, Calculate Vacancy: {calculate_vacancy}\n"

            pipeline_results: List[Dict[str, Any]] = []
            run_output_dir_relative_path: Optional[Path] = None # Relative to temp_output_base
            processing_error: Optional[str] = None

            try:
                pipeline_results, run_output_dir_relative_path = run_pdf_processing(
                    temp_input_pdf,
                    temp_output_base,
                    generate_images=generate_images,
                    calculate_vacancy=calculate_vacancy
                )

                if run_output_dir_relative_path is None:
                    processing_error = "Processing script failed to return an output directory name."
                elif not pipeline_results and not list((temp_output_base / run_output_dir_relative_path).glob('*')):
                     processing_error = f"Processing finished, but no units were extracted and no output files were found in '{run_output_dir_relative_path}'. Check PDF content and API key/quota."
                elif not pipeline_results:
                     status_message += f"Processing finished for {temp_input_pdf.name}, but no specific unit data was extracted. Output files might still be generated.\n"
                else:
                    # --- Copy CSVs to Persistent Cache, Populate Map, Load First DF ---
                    status_message += "\nCopying results to preview cache...\n"
                    unit_names = []
                    first_unit_name = None
                    run_output_dir_full_path_source = temp_output_base # Base for resolving source paths

                    for i, unit_result in enumerate(pipeline_results):
                        unit_name = unit_result.get('unit_name', f'Unit_{i+1}')
                        relative_csv_path = unit_result.get('csv_path')

                        if relative_csv_path:
                            original_full_csv_path = (run_output_dir_full_path_source / relative_csv_path).resolve()
                            # Use only the filename for the destination path within the cache dir
                            copied_csv_path = (preview_cache_dir / relative_csv_path.name).resolve()

                            if original_full_csv_path.exists():
                                try:
                                    shutil.copy(original_full_csv_path, copied_csv_path)
                                    # *** Store the path to the COPIED file in the map ***
                                    unit_data_map[unit_name] = copied_csv_path
                                    unit_names.append(unit_name)
                                    logging.debug(f"Copied preview for '{unit_name}' to: {copied_csv_path}")
                                    if first_unit_name is None:
                                        first_unit_name = unit_name
                                except Exception as copy_err:
                                    logging.error(f"Failed to copy preview CSV for '{unit_name}' from {original_full_csv_path} to {copied_csv_path}: {copy_err}", exc_info=True)
                                    status_message += f"- Warning: Failed to copy preview for {unit_name}.\n"
                            else:
                                 logging.warning(f"Source CSV not found for unit '{unit_name}' at {original_full_csv_path}, cannot copy to cache.")
                                 status_message += f"- Warning: Source CSV not found for {unit_name}, cannot cache preview.\n"
                        else:
                            status_message += f"- Warning: No CSV path found for unit result {i}. Cannot add to dropdown or cache.\n"

                    # --- Load Initial Preview (from cache) ---
                    if first_unit_name and first_unit_name in unit_data_map:
                         status_message += f"Attempting to load initial preview for: {first_unit_name} (from cache)\n"
                         # Get the path to the copied file directly from the map
                         first_copied_csv_path = unit_data_map[first_unit_name]
                         logging.info(f"Reading initial cached CSV from: {first_copied_csv_path}")
                         preview_label = f"Preview: {first_unit_name} ({first_copied_csv_path.name})"
                         try:
                            if first_copied_csv_path.exists() and first_copied_csv_path.stat().st_size > 0:
                                df_preview_data = pl.read_csv(first_copied_csv_path, try_parse_dates=True)
                                first_unit_df_data = gr.update(value=df_preview_data, label=preview_label)
                                df_group_update = gr.update(visible=True) # Show group
                                unit_selector_update = gr.update(choices=unit_names, value=first_unit_name, visible=True) # Show dropdown
                                status_message += f"- Successfully loaded initial preview for {first_unit_name}.\n"
                            else:
                                status_message += f"- Initial preview failed: Cached CSV for {first_unit_name} not found or empty at {first_copied_csv_path}.\n"
                         except Exception as preview_err:
                             logging.error(f"Error generating initial DataFrame preview from {first_copied_csv_path}: {preview_err}", exc_info=True)
                             status_message += f"- Error reading initial cached CSV for {first_unit_name}: {preview_err}\n"
                    elif unit_names: # Units exist but couldn't cache/load first one?
                         status_message += "- Units found, but couldn't load initial preview (copy or read error?). Check logs.\n"
                    else: # No units with CSVs found or copied
                        status_message += "- No units with CSV data found or cached to display.\n"

            except Exception as e:
                logging.error(f"Error during PDF processing logic: {e}", exc_info=True)
                processing_error = f"An unexpected error occurred during processing: {e}"

            # --- Handle processing outcome ---
            if processing_error:
                status_message += f"Error: {processing_error}\n"
                logging.error(status_message)
                # Return error state (8 items)
                return (None, status_message, None,
                        gr.update(value=None), gr.update(visible=False),
                        gr.update(choices=[], value=None, visible=False), {},
                        preview_cache_dir_state_update) # Keep cache dir path for potential cleanup later if needed

            # --- Zip the results (from original temp location) ---
            if run_output_dir_relative_path:
                run_output_dir_full_path_source = temp_output_base / run_output_dir_relative_path
                if not run_output_dir_full_path_source.is_dir() or not any(run_output_dir_full_path_source.iterdir()):
                    status_message += f"Warning: Original output directory '{run_output_dir_full_path_source}' not found or empty. Cannot create zip.\n"
                    final_zip_path = None
                    persistent_zip_path_state = None
                else:
                    zip_filename = f"{run_output_dir_relative_path.stem}_output.zip"
                    zip_filepath = temp_dir / zip_filename # Zip inside the inner temp dir
                    try:
                        logging.info(f"Creating zip file: {zip_filepath} from directory {run_output_dir_full_path_source}")
                        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            for item in run_output_dir_full_path_source.rglob('*'):
                                if item.is_file():
                                    arcname = item.relative_to(run_output_dir_full_path_source)
                                    zipf.write(item, arcname=arcname)

                        status_message += f"Successfully created zip file: {zip_filename}\n"
                        logging.info(status_message)

                        # --- Copy zip to a persistent location with the desired name ---
                        persistent_final_zip_path = Path(tempfile.gettempdir()) / zip_filename
                        try:
                            shutil.copy(zip_filepath, persistent_final_zip_path)
                            logging.info(f"Copied final zip to persistent temp path: {persistent_final_zip_path}")
                            # Convert Path object to string for Gradio state and output
                            final_zip_path = str(persistent_final_zip_path)
                            persistent_zip_path_state = str(persistent_final_zip_path)
                        except Exception as copy_final_err:
                             logging.error(f"Error copying final zip from {zip_filepath} to {persistent_final_zip_path}: {copy_final_err}", exc_info=True)
                             status_message += f"Error: Failed to copy final zip file: {copy_final_err}\n"
                             final_zip_path = None
                             persistent_zip_path_state = None
                    except Exception as e:
                        logging.error(f"Error creating zip file {zip_filepath}: {e}", exc_info=True)
                        status_message += f"Error: Failed to create zip file: {e}\n"
                        final_zip_path = None
                        persistent_zip_path_state = None
            else:
                 status_message += "Skipping zip creation as processing might have failed early.\n"
                 final_zip_path = None
                 persistent_zip_path_state = None

        # --- Inner temp directory is now cleaned up by the 'with' block ---

        # --- Prepare final return tuple (8 items) ---
        return (final_zip_path, status_message, persistent_zip_path_state,
                first_unit_df_data, df_group_update, unit_selector_update, unit_data_map,
                preview_cache_dir_state_update)

    except Exception as e:
        # This catches errors outside the inner 'with' block, like the mkdtemp failure
        # or potentially errors after the inner block finishes but before return
        logging.error(f"Error during outer processing or setup/cleanup: {e}", exc_info=True)
        status_message = f"Error during processing: {e}"
        # Try to clean up cache dir if it was created
        if preview_cache_dir and preview_cache_dir.exists():
             try:
                 shutil.rmtree(preview_cache_dir)
                 logging.info(f"Cleaned up preview cache directory {preview_cache_dir} due to outer error.")
             except Exception as cleanup_err:
                 logging.error(f"Failed to cleanup preview cache directory {preview_cache_dir} after error: {cleanup_err}")
        # Return error state (8 items)
        return (None, status_message, None,
                gr.update(value=None), gr.update(visible=False),
                gr.update(choices=[], value=None, visible=False), {},
                None)

# --- Dropdown Change Handler ---
def update_df_preview(selected_unit: str, unit_data_map: Dict[str, Path]) -> Dict:
    """
    Called when the unit selection dropdown changes. Loads and displays the selected unit's CSV.
    Uses the map which now stores FULL paths to files in the persistent cache.
    """
    if not selected_unit or not unit_data_map:
        logging.warning("Dropdown changed but missing selection or map. Cannot update preview.")
        return gr.update(value=None, label="DataFrame Preview (Error: Missing data)")

    full_csv_path = unit_data_map.get(selected_unit)

    if not full_csv_path or not isinstance(full_csv_path, Path):
         logging.error(f"Invalid or missing path stored for unit '{selected_unit}'. Map content: {unit_data_map}")
         return gr.update(value=None, label=f"DataFrame Preview (Error: Bad path for '{selected_unit}')")

    logging.info(f"Dropdown changed to: {selected_unit}. Attempting to load from cache: {full_csv_path}")

    try:
        if full_csv_path.exists() and full_csv_path.stat().st_size > 0:
            df = pl.read_csv(full_csv_path, try_parse_dates=True)
            preview_label = f"Preview: {selected_unit} ({full_csv_path.name})"
            logging.info(f"Updated preview to show {selected_unit}")
            return gr.update(value=df, label=preview_label)
        elif not full_csv_path.exists():
             logging.warning(f"Cached CSV file for selected unit '{selected_unit}' not found at: {full_csv_path}")
             return gr.update(value=None, label=f"DataFrame Preview (Error: File not found for '{selected_unit}')")
        else: # File is empty
             logging.warning(f"Cached CSV file for selected unit '{selected_unit}' is empty at: {full_csv_path}")
             return gr.update(value=None, label=f"DataFrame Preview (Error: File empty for '{selected_unit}')")
    except Exception as e:
        logging.error(f"Error reading cached CSV for {selected_unit} at {full_csv_path}: {e}", exc_info=True)
        return gr.update(value=None, label=f"DataFrame Preview (Error reading {selected_unit})")


# --- Reset Function ---
# Update signature: Add preview_cache_dir_state input
def reset_state(current_zip_path: Optional[str], preview_cache_dir: Optional[str]) -> Tuple[
    None, str, None, None, # pdf_input, status_output, zip_output, zip_path_state
    Dict, Dict, Dict, None, # df_preview update, df_group update, unit_selector update, unit_data_map_state clear
    None # preview_cache_dir_state clear
]: # Now returns 9 items total
    """
    Clears the UI elements, state, the persistent preview cache directory,
    and the named temporary zip file.
    """
    status = "State reset."
    zip_removed_status = ""
    cache_removed_status = ""

    # --- Clean up named temporary zip file ---
    if current_zip_path:
        zip_path = Path(current_zip_path)
        logging.info(f"Reset triggered. Attempting to remove temp zip file: {zip_path}")
        if zip_path.exists() and zip_path.is_file():
            try:
                zip_path.unlink() # Use unlink to remove the file
                logging.info(f"Successfully removed temporary zip file: {zip_path}")
                zip_removed_status = f" Removed temp file {zip_path.name}."
            except Exception as e:
                logging.error(f"Error removing temporary zip file {zip_path}: {e}", exc_info=True)
                zip_removed_status = f" Error removing temp file {zip_path.name}: {e}."
        else:
            logging.warning(f"Temporary zip file path found in state ({current_zip_path}), but file does not exist or is not a file.")
            zip_removed_status = f" Temp file {zip_path.name} not found."
    else:
        logging.info("Reset triggered. No temp zip file path was stored.")
        zip_removed_status = ""

    # --- Clean up persistent preview cache directory ---
    if preview_cache_dir:
        cache_path = Path(preview_cache_dir)
        if cache_path.exists() and cache_path.is_dir():
            try:
                shutil.rmtree(cache_path)
                logging.info(f"Successfully removed preview cache directory: {cache_path}")
                status += f" Removed preview cache."
            except Exception as e:
                logging.error(f"Error removing preview cache directory {cache_path}: {e}", exc_info=True)
                status += f" Error removing preview cache: {e}."
        else:
            logging.warning(f"Preview cache directory path found in state ({preview_cache_dir}), but directory does not exist or is not a directory.")
            cache_removed_status = " Preview cache not found."
    else:
        logging.info("No preview cache directory path found in state to remove.")
        cache_removed_status = ""

    # Combine status messages
    final_status = f"State reset.{zip_removed_status}{cache_removed_status}"

    # Clear the DataFrame, hide the results group, reset the dropdown, clear map state
    df_update = gr.update(value=None, label="DataFrame Preview")
    group_update = gr.update(visible=False)
    dropdown_update = gr.update(choices=[], value=None, visible=False)
    map_clear = None # Returning None clears State
    cache_dir_clear = None # Returning None clears State

    # 9 items: pdf_input, status_output, zip_output, zip_path_state, df_preview, df_results_group, unit_selector_dd, unit_data_map_state, preview_cache_dir_state
    return None, final_status, None, None, df_update, group_update, dropdown_update, map_clear, cache_dir_clear


# --- Gradio Interface Definition ---

with gr.Blocks(title="DHCR PDF Parser & Tools") as demo:
    gr.Markdown("# DHCR Tools")

    with gr.Tabs():
        with gr.TabItem("PDF Parser"): # --- TAB 1: PDF Parser ---
            gr.Markdown("## PDF Parser")
            gr.Markdown(
                "Upload a DHCR Rent History PDF file. The script will attempt to extract unit data using AI, "
                "optionally calculate vacancy allowances, generate images, and provide a downloadable zip file with the results."
                "\n**Note:** Processing can take several minutes depending on the PDF size and API response times."
                "\n**Requires a `GEMINI_API_KEY` environment variable or a `.env` file.**")

            # --- Original PDF Parser Layout START ---
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
                    with gr.Row():
                        images_checkbox = gr.Checkbox(label="Generate Page Images", value=True)
                        vacancy_checkbox = gr.Checkbox(label="Calculate Vacancy Allowance (Alpha)", value=False)
                    submit_button = gr.Button("Process PDF", variant="primary")
                    reset_button = gr.Button("Reset")
                with gr.Column(scale=2):
                    status_output = gr.Textbox(label="Status / Logs", lines=8, interactive=False)
                    # Group for results area (Dropdown + DataFrame)
                    with gr.Group(visible=False) as df_results_group:
                        unit_selector_dd = gr.Dropdown(label="Select Unit to Preview", interactive=True, visible=False)
                        df_preview = gr.DataFrame(label="DataFrame Preview", wrap=True)

                    zip_output = gr.File(label="Download Results (.zip)")
                    # State components need to be accessible by handlers, define them outside the tab if needed, but usually fine here
                    zip_path_state = gr.State(value=None)
                    unit_data_map_state = gr.State(value={}) # Holds map {unit_name: full_copied_csv_path}
                    preview_cache_dir_state = gr.State(value=None) # Holds path to persistent cache dir
            # --- Original PDF Parser Layout END ---

            # --- PDF Parser Event Handling (Remains associated with components above) ---
            submit_button.click(
                fn=process_dhcr_pdf,
                inputs=[pdf_input, images_checkbox, vacancy_checkbox],
                outputs=[
                    zip_output, status_output, zip_path_state,
                    df_preview, df_results_group, unit_selector_dd, unit_data_map_state,
                    preview_cache_dir_state
                ]
            )

            unit_selector_dd.change(
                fn=update_df_preview,
                inputs=[unit_selector_dd, unit_data_map_state],
                outputs=[df_preview]
            )

            reset_button.click(
                fn=reset_state,
                inputs=[zip_path_state, preview_cache_dir_state],
                outputs=[
                    pdf_input, status_output, zip_output, zip_path_state,
                    df_preview, df_results_group, unit_selector_dd, unit_data_map_state,
                    preview_cache_dir_state
                ]
            )
        # --- END TAB 1 ---

        with gr.TabItem("Vacancy Allowance Calculator"): # --- TAB 2: Vacancy Allowance Calculator ---
             # Render the calculator tab UI built in the other file
             # The create_calculator_tab() function returns a gr.Blocks instance
             # which needs to be rendered within this TabItem context.
             # We need to call the function *inside* the TabItem context.
             if calculator_tab_available:
                  calculator_ui = create_calculator_tab()
                  # If create_calculator_tab returns a Blocks object, render it implicitly
                  # No explicit render() call needed if Blocks is created within context
             else:
                  gr.Markdown("## Vacancy AllowanceCalculator")
                  gr.Markdown("**Error:** Failed to load the calculator tab module. Please check the application logs.")
        # --- END TAB 2 ---

# --- Launch the App ---
if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
         logging.warning("GEMINI_API_KEY not found in environment variables or .env file.")
         print("WARNING: GEMINI_API_KEY environment variable not set.")
         print("The application requires this key to function.")
         print("Please set it in your environment or create a '.env' file with GEMINI_API_KEY=your_key")

    print("Launching Gradio App...")
    # Consider adding queue() if processing takes time
    # demo.queue()
    demo.launch() 