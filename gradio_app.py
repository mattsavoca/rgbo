import gradio as gr
import tempfile
import zipfile
from pathlib import Path
import shutil
import logging
import os
from typing import Optional, Tuple, Any, List, Dict, Union
import polars as pl # Use polars for reading CSVs

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [GRADIO_APP] - %(message)s')

# Import the calculator tab creator FIRST (after logging config)
try:
    from tabs.vacancy_calculator_tab import create_calculator_tab
    # Use logging directly here as it's configured just above
    logging.info("Successfully imported UI builder from vacancy_calculator_tab.")
    calculator_tab_available = True
except ImportError as e:
    logging.error(f"Failed to import vacancy_calculator_tab: {e}. Calculator tab will be disabled.", exc_info=True)
    calculator_tab_available = False
    # Define a dummy function if import fails
    def create_calculator_tab(*args):
        with gr.Blocks() as calculator_tab_error:
            gr.Markdown("## Vacancy Allowance Calculator")
            gr.Markdown("**Error:** Failed to load the calculator tab module. Please check the application logs.")
        return calculator_tab_error

# --- Import Core Logic ---
# DHCR PDF Parser Logic
try:
    # Check if API key is already in environment (e.g., from run.sh sourcing)
    if os.environ.get("GEMINI_API_KEY"):
        logging.info("GEMINI_API_KEY found in environment variables.")
    else:
        # If not in env, try loading from .env file as a fallback
        logging.info("GEMINI_API_KEY not found in environment. Attempting to load from .env file.")
        try:
            from dotenv import load_dotenv
            # Load from default .env path
            if load_dotenv(): 
                logging.info("Successfully loaded environment variables from .env file.")
            else:
                logging.warning(".env file not found or empty. Proceeding without loading from .env.")
        except ImportError:
            logging.warning("'python-dotenv' package not found. Cannot load .env file. Relying on existing environment variables.")
        # Check again after attempting to load .env
        if not os.environ.get("GEMINI_API_KEY"):
            logging.warning("GEMINI_API_KEY still not found after checking environment and .env file.")
            # NOTE: The app might fail later if the key is required but not found
    
    # Now proceed with imports that might depend on the API key
    from scripts.pdf_handler import run_pdf_processing
    logging.info("Successfully imported run_pdf_processing from pdf_handler.")
except ImportError as e:
    logging.error(f"Fatal Error: Failed to import DHCR processing logic (pdf_handler): {e}", exc_info=True)
    def run_pdf_processing(*args, **kwargs):
        logging.error("run_pdf_processing function is unavailable due to import error.")
        raise RuntimeError("Core DHCR PDF processing logic failed to load. Cannot continue.") from e
except Exception as e:
     logging.error(f"An unexpected error occurred during DHCR import or env checking: {e}", exc_info=True)
     def run_pdf_processing(*args, **kwargs):
        logging.error("run_pdf_processing function is unavailable due to an unexpected setup error.")
        raise RuntimeError("Core DHCR PDF processing logic failed to load. Cannot continue.") from e

# CBB PDF Parser Logic
try:
    # No need to check/load env again, assume it was handled above
    from scripts.cbb_handler import run_cbb_processing
    logging.info("Successfully imported run_cbb_processing from cbb_handler.")
except ImportError as e:
    logging.error(f"Fatal Error: Failed to import CBB processing logic (cbb_handler): {e}", exc_info=True)
    # Define a dummy function to avoid app crash if CBB handler is missing
    def run_cbb_processing(*args, **kwargs):
        logging.error("run_cbb_processing function is unavailable due to import error.")
        raise RuntimeError("Core CBB PDF processing logic failed to load. CBB tab will not function.") from e
except Exception as e:
     logging.error(f"An unexpected error occurred during CBB import: {e}", exc_info=True)
     def run_cbb_processing(*args, **kwargs):
        logging.error("run_cbb_processing function is unavailable due to an unexpected setup error.")
        raise RuntimeError("Core CBB PDF processing logic failed to load. CBB tab will not function.") from e

# --- Gradio Processing Function (DHCR) ---

# Update return signature: Add preview_cache_dir_state output AND processed_data_state output
def process_dhcr_pdf(pdf_file_obj, generate_images: bool, calculate_vacancy: bool) -> Tuple[
    Optional[str], str, Optional[str], # zip_output, status_output, zip_path_state
    Dict, Dict, Dict, Dict, # df_preview update, df_group update, unit_selector update, unit_data_map_state update
    Optional[str], # preview_cache_dir_state update
    Dict[str, pl.DataFrame] # processed_data_state update
]: # Now returns 9 items
    """
    Main function called by Gradio interface for DHCR PDF Processing.
    Processes PDF, copies preview CSVs to a persistent cache, zips results,
    prepares data for the first unit's DataFrame, populates unit selector dropdown,
    stores a map of unit names to FULL CSV paths in the persistent cache,
    and loads all unit dataframes into a dictionary for the calculator tab.
    """
    # Initialize return values
    first_unit_df_data = gr.update(value=None)
    df_group_update = gr.update(visible=False)
    unit_selector_update = gr.update(choices=[], value=None, visible=False)
    unit_data_map: Dict[str, Path] = {} # Map unit name to its FULL CSV path IN CACHE
    processed_data_dict: Dict[str, pl.DataFrame] = {} # Map unit name to its DataFrame (for Tab 2)
    final_zip_path: Optional[str] = None
    persistent_zip_path_state: Optional[str] = None
    preview_cache_dir: Optional[Path] = None # Path to the persistent preview dir
    preview_cache_dir_state_update: Optional[str] = None

    if pdf_file_obj is None:
        return (None, "Error: No PDF file provided.", None,
                first_unit_df_data, df_group_update, unit_selector_update, unit_data_map,
                preview_cache_dir_state_update, processed_data_dict)

    uploaded_pdf_path = Path(pdf_file_obj.name)
    logging.info(f"[DHCR] Received file: {uploaded_pdf_path.name}, Generate Images: {generate_images}, Calculate Vacancy: {calculate_vacancy}")

    # --- Create Persistent Preview Cache Directory FIRST ---
    try:
        # Using a more descriptive prefix
        preview_cache_dir = Path(tempfile.mkdtemp(prefix="gradio_dhcr_preview_"))
        preview_cache_dir_state_update = str(preview_cache_dir) # Store path for state return
        logging.info(f"[DHCR] Created persistent preview cache directory: {preview_cache_dir}")
    except Exception as e:
        logging.error(f"[DHCR] Failed to create persistent preview cache directory: {e}", exc_info=True)
        status_message = f"Fatal Error: Could not create preview cache directory: {e}"
        # Return error state (9 items)
        return (None, status_message, None,
                gr.update(value=None), gr.update(visible=False),
                gr.update(choices=[], value=None, visible=False), {}, None, {})

    try:
        # --- Process PDF in its own temporary directory ---
        with tempfile.TemporaryDirectory(prefix="dhcr_proc_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            temp_input_pdf = temp_dir / uploaded_pdf_path.name
            # This is where run_pdf_processing will create output/<pdf_stem>
            temp_output_base = temp_dir / "output"
            temp_output_base.mkdir()
            shutil.copy(uploaded_pdf_path, temp_input_pdf)
            logging.info(f"[DHCR] Copied uploaded PDF to temporary input: {temp_input_pdf}")

            status_message = f"[DHCR] Processing {temp_input_pdf.name}...\n"
            status_message += f"Generate Images: {generate_images}, Calculate Vacancy: {calculate_vacancy}\n"

            pipeline_results: List[Dict[str, Any]] = []
            run_output_dir_relative_path: Optional[Path] = None # Relative to temp_output_base
            processing_error: Optional[str] = None

            try:
                # Call the DHCR-specific handler
                pipeline_results, run_output_dir_relative_path = run_pdf_processing(
                    temp_input_pdf,
                    temp_output_base,
                    generate_images=generate_images,
                    calculate_vacancy=calculate_vacancy
                )

                if run_output_dir_relative_path is None:
                    processing_error = "[DHCR] Processing script failed to return an output directory name."
                elif not pipeline_results and not list((temp_output_base / run_output_dir_relative_path).glob('*')):
                     processing_error = f"[DHCR] Processing finished, but no units were extracted and no output files were found in '{run_output_dir_relative_path}'. Check PDF content and API key/quota."
                elif not pipeline_results:
                     status_message += f"[DHCR] Processing finished for {temp_input_pdf.name}, but no specific unit data was extracted. Output files might still be generated.\n"
                else:
                    # --- Copy CSVs to Persistent Cache, Populate Map, Load First DF ---
                    status_message += "\n[DHCR] Copying results to preview cache...\n"
                    unit_names = []
                    first_unit_name = None
                    run_output_dir_full_path_source = temp_output_base # Base for resolving source paths

                    for i, unit_result in enumerate(pipeline_results):
                        # DHCR handler returns 'unit_name' and 'csv_path' (relative)
                        unit_name = unit_result.get('unit_name', f'Unit_{i+1}')
                        relative_csv_path = unit_result.get('csv_path') # This is Path('pdf_stem/apt_XYZ.csv')

                        if relative_csv_path and isinstance(relative_csv_path, Path):
                            original_full_csv_path = (run_output_dir_full_path_source / relative_csv_path).resolve()
                            # Use only the filename for the destination path within the cache dir
                            copied_csv_path = (preview_cache_dir / relative_csv_path.name).resolve()

                            if original_full_csv_path.exists():
                                try:
                                    shutil.copy(original_full_csv_path, copied_csv_path)
                                    # Store the path to the COPIED file in the map
                                    unit_data_map[unit_name] = copied_csv_path
                                    unit_names.append(unit_name)
                                    logging.debug(f"[DHCR] Copied preview for '{unit_name}' to: {copied_csv_path}")
                                    if first_unit_name is None:
                                        first_unit_name = unit_name
                                except Exception as copy_err:
                                    logging.error(f"[DHCR] Failed to copy preview CSV for '{unit_name}' from {original_full_csv_path} to {copied_csv_path}: {copy_err}", exc_info=True)
                                    status_message += f"- Warning: Failed to copy preview for {unit_name}.\n"
                            else:
                                 logging.warning(f"[DHCR] Source CSV not found for unit '{unit_name}' at {original_full_csv_path}, cannot copy to cache.")
                                 status_message += f"- Warning: Source CSV not found for {unit_name}, cannot cache preview.\n"
                        else:
                            status_message += f"- Warning: No valid CSV path found for DHCR result {i}. Cannot add to dropdown or cache.\n"

                    # --- Load Initial Preview (from cache) ---
                    if first_unit_name and first_unit_name in unit_data_map:
                         status_message += f"[DHCR] Attempting to load initial preview for: {first_unit_name} (from cache)\n"
                         first_copied_csv_path = unit_data_map[first_unit_name]
                         logging.info(f"[DHCR] Reading initial cached CSV from: {first_copied_csv_path}")
                         preview_label = f"Preview: {first_unit_name} ({first_copied_csv_path.name})"
                         try:
                            if first_copied_csv_path.exists() and first_copied_csv_path.stat().st_size > 0:
                                df_preview_data = pl.read_csv(first_copied_csv_path, try_parse_dates=True)
                                first_unit_df_data = gr.update(value=df_preview_data, label=preview_label)
                                df_group_update = gr.update(visible=True) # Show group
                                unit_selector_update = gr.update(choices=sorted(unit_names), value=first_unit_name, visible=True) # Show dropdown, sorted
                                status_message += f"- Successfully loaded initial preview for {first_unit_name}.\n"
                            else:
                                status_message += f"- Initial preview failed: Cached CSV for {first_unit_name} not found or empty at {first_copied_csv_path}.\n"
                         except Exception as preview_err:
                             logging.error(f"[DHCR] Error generating initial DataFrame preview from {first_copied_csv_path}: {preview_err}", exc_info=True)
                             status_message += f"- Error reading initial cached CSV for {first_unit_name}: {preview_err}\n"
                    elif unit_names:
                         status_message += "- Units found, but couldn't load initial preview (copy or read error?). Check logs.\n"
                         unit_selector_update = gr.update(choices=sorted(unit_names), value=None, visible=True) # Show dropdown anyway
                    else:
                        status_message += "- No units with CSV data found or cached to display.\n"
                        unit_selector_update = gr.update(choices=[], value=None, visible=False)

                    # --- Load ALL Unit DataFrames into processed_data_dict (for Tab 2) ---
                    status_message += "\n[DHCR] Loading all unit data for calculator tab preview...\n"
                    for unit_name_load, cached_csv_path_load in unit_data_map.items():
                        try:
                            if cached_csv_path_load.exists() and cached_csv_path_load.stat().st_size > 0:
                                df = pl.read_csv(cached_csv_path_load, try_parse_dates=True)
                                processed_data_dict[unit_name_load] = df
                                logging.info(f"[DHCR] Loaded DataFrame for '{unit_name_load}' into shared state.")
                                status_message += f"- Loaded data for {unit_name_load}.\n"
                            else:
                                 logging.warning(f"[DHCR] Skipping DF load for calc tab preview for '{unit_name_load}': CSV not found/empty at {cached_csv_path_load}")
                                 status_message += f"- Warning: Could not load data for {unit_name_load} (file missing or empty).\n"
                        except Exception as load_err:
                            logging.error(f"[DHCR] Failed to load DF for calc tab preview for '{unit_name_load}' from {cached_csv_path_load}: {load_err}")
                            status_message += f"- Error: Failed to load data for {unit_name_load}.\n"
                    status_message += "[DHCR] Finished loading data for calculator tab.\n"
                    # --- End Load ALL ---

            except RuntimeError as rt_err:
                 # Catch runtime errors specifically from the handler itself (e.g., import issues)
                 logging.error(f"[DHCR] Runtime Error during processing logic: {rt_err}", exc_info=True)
                 processing_error = f"A critical error occurred in the DHCR processing module: {rt_err}"
            except Exception as e:
                logging.error(f"[DHCR] Error during PDF processing logic: {e}", exc_info=True)
                processing_error = f"An unexpected error occurred during DHCR processing: {e}"

            # --- Handle processing outcome ---
            if processing_error:
                status_message += f"Error: {processing_error}\n"
                logging.error(status_message)
                # Return error state (9 items)
                return (None, status_message, None,
                        gr.update(value=None), gr.update(visible=False),
                        gr.update(choices=[], value=None, visible=False), {}, preview_cache_dir_state_update, {})

            # --- Zip the results (from original temp location) ---
            if run_output_dir_relative_path:
                run_output_dir_full_path_source = temp_output_base / run_output_dir_relative_path
                if not run_output_dir_full_path_source.is_dir() or not any(run_output_dir_full_path_source.iterdir()):
                    status_message += f"Warning: Original output directory '{run_output_dir_full_path_source}' not found or empty. Cannot create zip.\n"
                    final_zip_path = None
                    persistent_zip_path_state = None
                else:
                    zip_filename = f"{run_output_dir_relative_path.stem}_DHCR_output.zip"
                    zip_filepath = temp_dir / zip_filename # Zip inside the inner temp dir
                    try:
                        logging.info(f"[DHCR] Creating zip file: {zip_filepath} from directory {run_output_dir_full_path_source}")
                        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            for item in run_output_dir_full_path_source.rglob('*'):
                                if item.is_file():
                                    arcname = item.relative_to(run_output_dir_full_path_source)
                                    zipf.write(item, arcname=arcname)

                        status_message += f"Successfully created zip file: {zip_filename}\n"
                        logging.info(status_message)

                        # Copy zip to a persistent location
                        persistent_final_zip_path = Path(tempfile.gettempdir()) / zip_filename
                        try:
                            shutil.copy(zip_filepath, persistent_final_zip_path)
                            logging.info(f"[DHCR] Copied final zip to persistent temp path: {persistent_final_zip_path}")
                            final_zip_path = str(persistent_final_zip_path)
                            persistent_zip_path_state = str(persistent_final_zip_path)
                        except Exception as copy_final_err:
                             logging.error(f"[DHCR] Error copying final zip from {zip_filepath} to {persistent_final_zip_path}: {copy_final_err}", exc_info=True)
                             status_message += f"Error: Failed to copy final zip file: {copy_final_err}\n"
                             final_zip_path = None
                             persistent_zip_path_state = None
                    except Exception as e:
                        logging.error(f"[DHCR] Error creating zip file {zip_filepath}: {e}", exc_info=True)
                        status_message += f"Error: Failed to create zip file: {e}\n"
                        final_zip_path = None
                        persistent_zip_path_state = None
            else:
                 status_message += "[DHCR] Skipping zip creation as processing might have failed early.\n"
                 final_zip_path = None
                 persistent_zip_path_state = None

        # --- Inner temp directory is now cleaned up by the 'with' block ---

        # --- Prepare final return tuple (9 items) ---
        return (final_zip_path, status_message, persistent_zip_path_state,
                first_unit_df_data, df_group_update, unit_selector_update, unit_data_map,
                preview_cache_dir_state_update, processed_data_dict)

    except Exception as e:
        # Catches errors outside the inner 'with' block
        logging.error(f"[DHCR] Error during outer processing or setup/cleanup: {e}", exc_info=True)
        status_message = f"Error during DHCR processing: {e}"
        if preview_cache_dir and preview_cache_dir.exists():
             try:
                 shutil.rmtree(preview_cache_dir)
                 logging.info(f"[DHCR] Cleaned up preview cache directory {preview_cache_dir} due to outer error.")
             except Exception as cleanup_err:
                 logging.error(f"[DHCR] Failed to cleanup preview cache directory {preview_cache_dir} after error: {cleanup_err}")
        # Return error state (9 items)
        return (None, status_message, None,
                gr.update(value=None), gr.update(visible=False),
                gr.update(choices=[], value=None, visible=False), {}, None, {})

# --- Dropdown Change Handler (DHCR) ---
def update_df_preview(selected_unit: str, unit_data_map: Dict[str, Path]) -> Dict:
    """
    Called when the DHCR unit selection dropdown changes.
    Uses the map storing FULL paths to files in the persistent DHCR cache.
    """
    if not selected_unit or not unit_data_map:
        logging.warning("[DHCR] Dropdown changed but missing selection or map. Cannot update preview.")
        return gr.update(value=None, label="DataFrame Preview (Error: Missing data)")

    full_csv_path = unit_data_map.get(selected_unit)

    if not full_csv_path or not isinstance(full_csv_path, Path):
         logging.error(f"[DHCR] Invalid or missing path stored for unit '{selected_unit}'. Map content: {unit_data_map}")
         return gr.update(value=None, label=f"DataFrame Preview (Error: Bad path for '{selected_unit}')")

    logging.info(f"[DHCR] Dropdown changed to: {selected_unit}. Attempting to load from cache: {full_csv_path}")

    try:
        if full_csv_path.exists() and full_csv_path.stat().st_size > 0:
            df = pl.read_csv(full_csv_path, try_parse_dates=True)
            preview_label = f"Preview: {selected_unit} ({full_csv_path.name})"
            logging.info(f"[DHCR] Updated preview to show {selected_unit}")
            return gr.update(value=df, label=preview_label)
        elif not full_csv_path.exists():
             logging.warning(f"[DHCR] Cached CSV file for selected unit '{selected_unit}' not found at: {full_csv_path}")
             return gr.update(value=None, label=f"DataFrame Preview (Error: File not found for '{selected_unit}')")
        else: # File is empty
             logging.warning(f"[DHCR] Cached CSV file for selected unit '{selected_unit}' is empty at: {full_csv_path}")
             return gr.update(value=None, label=f"DataFrame Preview (Error: File empty for '{selected_unit}')")
    except Exception as e:
        logging.error(f"[DHCR] Error reading cached CSV for {selected_unit} at {full_csv_path}: {e}", exc_info=True)
        return gr.update(value=None, label=f"DataFrame Preview (Error reading {selected_unit})")

# --- Reset Function (DHCR) ---
# Update signature: Add preview_cache_dir_state input AND processed_units_data_state input
def reset_state(
    current_zip_path: Optional[str],
    preview_cache_dir: Optional[str],
    # processed_data: Dict # No longer needed as input
) -> Tuple[
    None, str, None, None, # pdf_input, status_output, zip_output, zip_path_state
    Dict, Dict, Dict, None, # df_preview update, df_group update, unit_selector update, unit_data_map_state clear
    None, # preview_cache_dir_state clear
    None # processed_units_data_state clear
]: # Now returns 10 items total
    """
    Clears the DHCR tab UI elements, state, the persistent preview cache directory,
    the named temporary zip file, and the shared processed data state.
    """
    status = "[DHCR] State reset."
    zip_removed_status = ""
    cache_removed_status = ""

    # --- Clean up named temporary zip file ---
    if current_zip_path:
        zip_path = Path(current_zip_path)
        logging.info(f"[DHCR] Reset triggered. Attempting to remove temp zip file: {zip_path}")
        if zip_path.exists() and zip_path.is_file():
            try:
                zip_path.unlink()
                logging.info(f"[DHCR] Successfully removed temporary zip file: {zip_path}")
                zip_removed_status = f" Removed temp file {zip_path.name}."
            except Exception as e:
                logging.error(f"[DHCR] Error removing temporary zip file {zip_path}: {e}", exc_info=True)
                zip_removed_status = f" Error removing temp file {zip_path.name}: {e}."
        else:
            logging.warning(f"[DHCR] Temporary zip file path found in state ({current_zip_path}), but file does not exist or is not a file.")
            zip_removed_status = f" Temp file {zip_path.name} not found."
    else:
        logging.info("[DHCR] Reset triggered. No temp zip file path was stored.")

    # --- Clean up persistent preview cache directory ---
    if preview_cache_dir:
        cache_path = Path(preview_cache_dir)
        if cache_path.exists() and cache_path.is_dir():
            try:
                shutil.rmtree(cache_path)
                logging.info(f"[DHCR] Successfully removed preview cache directory: {cache_path}")
                cache_removed_status = f" Removed preview cache {cache_path.name}."
            except Exception as e:
                logging.error(f"[DHCR] Error removing preview cache directory {cache_path}: {e}", exc_info=True)
                cache_removed_status = f" Error removing preview cache {cache_path.name}: {e}."
        else:
            logging.warning(f"[DHCR] Preview cache directory path found in state ({preview_cache_dir}), but directory does not exist or is not a directory.")
            cache_removed_status = f" Preview cache {cache_path.name} not found."
    else:
        logging.info("[DHCR] No preview cache directory path found in state to remove.")

    # Combine status messages
    final_status = f"[DHCR] State reset.{zip_removed_status}{cache_removed_status}"

    # Clear the DataFrame, hide the results group, reset the dropdown, clear map state
    df_update = gr.update(value=None, label="DataFrame Preview")
    group_update = gr.update(visible=False)
    dropdown_update = gr.update(choices=[], value=None, visible=False)
    map_clear = None
    cache_dir_clear = None
    processed_data_clear = None

    # 10 items: pdf_input, status_output, zip_output, zip_path_state, df_preview, df_results_group, unit_selector_dd, unit_data_map_state, preview_cache_dir_state, processed_units_data_state
    return None, final_status, None, None, df_update, group_update, dropdown_update, map_clear, cache_dir_clear, processed_data_clear

# --- Gradio Processing Function (CBB) ---

def process_cbb_pdf(pdf_file_obj, generate_images: bool) -> Tuple[
    Optional[str], str, Optional[str], # zip_output, status_output, zip_path_state
    Dict, Dict, Dict, Dict, # df_preview update, df_group update, docket_selector update, data_map_state update
    Optional[str] # preview_cache_dir_state update
]: # Returns 8 items
    """
    Main function called by Gradio interface for CBB PDF Processing.
    Processes PDF, copies preview CSVs to a persistent cache, zips results,
    prepares data for the first docket's DataFrame, populates docket selector dropdown,
    and stores a map of docket numbers to FULL CSV paths in the persistent cache.
    """
    # Initialize return values
    first_docket_df_data = gr.update(value=None)
    df_group_update = gr.update(visible=False)
    docket_selector_update = gr.update(choices=[], value=None, visible=False)
    docket_data_map: Dict[str, Path] = {} # Map docket number to its FULL CSV path IN CACHE
    final_zip_path: Optional[str] = None
    persistent_zip_path_state: Optional[str] = None
    preview_cache_dir: Optional[Path] = None # Path to the persistent preview dir
    preview_cache_dir_state_update: Optional[str] = None

    if pdf_file_obj is None:
        return (None, "Error: No PDF file provided.", None,
                first_docket_df_data, df_group_update, docket_selector_update, docket_data_map,
                preview_cache_dir_state_update)

    uploaded_pdf_path = Path(pdf_file_obj.name)
    logging.info(f"[CBB] Received file: {uploaded_pdf_path.name}, Generate Images: {generate_images}")

    # --- Create Persistent Preview Cache Directory FIRST ---
    try:
        preview_cache_dir = Path(tempfile.mkdtemp(prefix="gradio_cbb_preview_"))
        preview_cache_dir_state_update = str(preview_cache_dir)
        logging.info(f"[CBB] Created persistent preview cache directory: {preview_cache_dir}")
    except Exception as e:
        logging.error(f"[CBB] Failed to create persistent preview cache directory: {e}", exc_info=True)
        status_message = f"Fatal Error: Could not create preview cache directory: {e}"
        # Return error state (8 items)
        return (None, status_message, None,
                gr.update(value=None), gr.update(visible=False),
                gr.update(choices=[], value=None, visible=False), {}, None)

    try:
        # --- Process PDF in its own temporary directory ---
        with tempfile.TemporaryDirectory(prefix="cbb_proc_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            temp_input_pdf = temp_dir / uploaded_pdf_path.name
            temp_output_base = temp_dir / "output"
            temp_output_base.mkdir()
            shutil.copy(uploaded_pdf_path, temp_input_pdf)
            logging.info(f"[CBB] Copied uploaded PDF to temporary input: {temp_input_pdf}")

            status_message = f"[CBB] Processing {temp_input_pdf.name}...\n"
            status_message += f"Generate Images: {generate_images}\n"

            pipeline_results: List[Dict[str, Any]] = []
            run_output_dir_relative_path: Optional[Path] = None
            processing_error: Optional[str] = None

            try:
                # Call the CBB-specific handler
                pipeline_results, run_output_dir_relative_path = run_cbb_processing(
                    temp_input_pdf,
                    temp_output_base,
                    generate_images=generate_images
                )

                if run_output_dir_relative_path is None:
                    processing_error = "[CBB] Processing script failed to return an output directory name."
                # Check if the *directory* itself contains files, even if pipeline_results is empty (e.g., only images generated)
                output_dir_exists_and_has_content = (run_output_dir_relative_path and 
                                                     (temp_output_base / run_output_dir_relative_path).is_dir() and 
                                                     any((temp_output_base / run_output_dir_relative_path).iterdir()))

                if not pipeline_results and not output_dir_exists_and_has_content:
                     processing_error = f"[CBB] Processing finished, but no dockets were extracted and no output files (CSV or images) were found in '{run_output_dir_relative_path}'. Check PDF content and API key/quota."
                elif not pipeline_results:
                     status_message += f"[CBB] Processing finished for {temp_input_pdf.name}, but no specific docket data was extracted/reported. Output files (like images) might still be generated.\n"
                else:
                    # --- Copy CSVs to Persistent Cache, Populate Map, Load First DF ---
                    status_message += "\n[CBB] Copying results to preview cache...\n"
                    docket_nos = []
                    first_docket_no = None
                    run_output_dir_full_path_source = temp_output_base

                    for i, docket_result in enumerate(pipeline_results):
                        # CBB handler returns 'docket_no' and 'csv_path' (relative)
                        docket_no = docket_result.get('docket_no', f'Docket_{i+1}')
                        relative_csv_path = docket_result.get('csv_path') # Path('cbb_stem/docket_XYZ.csv')

                        if relative_csv_path and isinstance(relative_csv_path, Path):
                            original_full_csv_path = (run_output_dir_full_path_source / relative_csv_path).resolve()
                            # Use only the filename for destination in cache
                            copied_csv_path = (preview_cache_dir / relative_csv_path.name).resolve()

                            if original_full_csv_path.exists():
                                try:
                                    shutil.copy(original_full_csv_path, copied_csv_path)
                                    docket_data_map[docket_no] = copied_csv_path # Map docket_no -> copied path
                                    docket_nos.append(docket_no)
                                    logging.debug(f"[CBB] Copied preview for '{docket_no}' to: {copied_csv_path}")
                                    if first_docket_no is None:
                                        first_docket_no = docket_no
                                except Exception as copy_err:
                                    logging.error(f"[CBB] Failed to copy preview CSV for '{docket_no}' from {original_full_csv_path} to {copied_csv_path}: {copy_err}", exc_info=True)
                                    status_message += f"- Warning: Failed to copy preview for {docket_no}.\n"
                            else:
                                 logging.warning(f"[CBB] Source CSV not found for docket '{docket_no}' at {original_full_csv_path}, cannot copy to cache.")
                                 status_message += f"- Warning: Source CSV not found for {docket_no}, cannot cache preview.\n"
                        else:
                            status_message += f"- Warning: No valid CSV path found for CBB result {i}. Cannot add to dropdown or cache.\n"

                    # --- Load Initial Preview (from cache) ---
                    if first_docket_no and first_docket_no in docket_data_map:
                         status_message += f"[CBB] Attempting to load initial preview for: {first_docket_no} (from cache)\n"
                         first_copied_csv_path = docket_data_map[first_docket_no]
                         logging.info(f"[CBB] Reading initial cached CSV from: {first_copied_csv_path}")
                         preview_label = f"Preview: {first_docket_no} ({first_copied_csv_path.name})"
                         try:
                            if first_copied_csv_path.exists() and first_copied_csv_path.stat().st_size > 0:
                                df_preview_data = pl.read_csv(first_copied_csv_path, try_parse_dates=True)
                                first_docket_df_data = gr.update(value=df_preview_data, label=preview_label)
                                df_group_update = gr.update(visible=True)
                                docket_selector_update = gr.update(choices=sorted(docket_nos), value=first_docket_no, visible=True)
                                status_message += f"- Successfully loaded initial preview for {first_docket_no}.\n"
                            else:
                                status_message += f"- Initial preview failed: Cached CSV for {first_docket_no} not found or empty at {first_copied_csv_path}.\n"
                         except Exception as preview_err:
                             logging.error(f"[CBB] Error generating initial DataFrame preview from {first_copied_csv_path}: {preview_err}", exc_info=True)
                             status_message += f"- Error reading initial cached CSV for {first_docket_no}: {preview_err}\n"
                    elif docket_nos: # Dockets exist but couldn't cache/load first one
                         status_message += "- Dockets found, but couldn't load initial preview (copy or read error?). Check logs.\n"
                         docket_selector_update = gr.update(choices=sorted(docket_nos), value=None, visible=True)
                    else: # No dockets with CSVs found or copied
                        status_message += "- No dockets with CSV data found or cached to display.\n"
                        docket_selector_update = gr.update(choices=[], value=None, visible=False)

            except RuntimeError as rt_err:
                 logging.error(f"[CBB] Runtime Error during processing logic: {rt_err}", exc_info=True)
                 processing_error = f"A critical error occurred in the CBB processing module: {rt_err}"
            except Exception as e:
                logging.error(f"[CBB] Error during PDF processing logic: {e}", exc_info=True)
                processing_error = f"An unexpected error occurred during CBB processing: {e}"

            # --- Handle processing outcome ---
            if processing_error:
                status_message += f"Error: {processing_error}\n"
                logging.error(status_message)
                # Return error state (8 items)
                return (None, status_message, None,
                        gr.update(value=None), gr.update(visible=False),
                        gr.update(choices=[], value=None, visible=False), {}, preview_cache_dir_state_update)

            # --- Zip the results (from original temp location) ---
            if run_output_dir_relative_path:
                run_output_dir_full_path_source = temp_output_base / run_output_dir_relative_path
                # Check if the directory exists AND has *any* content (CSVs or images)
                if not run_output_dir_full_path_source.is_dir() or not any(run_output_dir_full_path_source.iterdir()):
                    status_message += f"Warning: Original output directory '{run_output_dir_full_path_source}' not found or empty. Cannot create zip.\n"
                    final_zip_path = None
                    persistent_zip_path_state = None
                else:
                    zip_filename = f"{run_output_dir_relative_path.stem}_CBB_output.zip"
                    zip_filepath = temp_dir / zip_filename
                    try:
                        logging.info(f"[CBB] Creating zip file: {zip_filepath} from directory {run_output_dir_full_path_source}")
                        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            for item in run_output_dir_full_path_source.rglob('*'):
                                if item.is_file():
                                    arcname = item.relative_to(run_output_dir_full_path_source)
                                    zipf.write(item, arcname=arcname)

                        status_message += f"Successfully created zip file: {zip_filename}\n"
                        logging.info(status_message)

                        # Copy zip to a persistent location
                        persistent_final_zip_path = Path(tempfile.gettempdir()) / zip_filename
                        try:
                            shutil.copy(zip_filepath, persistent_final_zip_path)
                            logging.info(f"[CBB] Copied final zip to persistent temp path: {persistent_final_zip_path}")
                            final_zip_path = str(persistent_final_zip_path)
                            persistent_zip_path_state = str(persistent_final_zip_path)
                        except Exception as copy_final_err:
                             logging.error(f"[CBB] Error copying final zip from {zip_filepath} to {persistent_final_zip_path}: {copy_final_err}", exc_info=True)
                             status_message += f"Error: Failed to copy final zip file: {copy_final_err}\n"
                             final_zip_path = None
                             persistent_zip_path_state = None
                    except Exception as e:
                        logging.error(f"[CBB] Error creating zip file {zip_filepath}: {e}", exc_info=True)
                        status_message += f"Error: Failed to create zip file: {e}\n"
                        final_zip_path = None
                        persistent_zip_path_state = None
            else:
                 status_message += "[CBB] Skipping zip creation as processing might have failed early.\n"
                 final_zip_path = None
                 persistent_zip_path_state = None

        # --- Inner temp directory cleaned up ---

        # --- Prepare final return tuple (8 items) ---
        return (final_zip_path, status_message, persistent_zip_path_state,
                first_docket_df_data, df_group_update, docket_selector_update, docket_data_map,
                preview_cache_dir_state_update)

    except Exception as e:
        logging.error(f"[CBB] Error during outer processing or setup/cleanup: {e}", exc_info=True)
        status_message = f"Error during CBB processing: {e}"
        if preview_cache_dir and preview_cache_dir.exists():
             try:
                 shutil.rmtree(preview_cache_dir)
                 logging.info(f"[CBB] Cleaned up preview cache directory {preview_cache_dir} due to outer error.")
             except Exception as cleanup_err:
                 logging.error(f"[CBB] Failed to cleanup preview cache directory {preview_cache_dir} after error: {cleanup_err}")
        # Return error state (8 items)
        return (None, status_message, None,
                gr.update(value=None), gr.update(visible=False),
                gr.update(choices=[], value=None, visible=False), {}, None)

# --- Dropdown Change Handler (CBB) ---
def update_cbb_df_preview(selected_docket: str, docket_data_map: Dict[str, Path]) -> Dict:
    """
    Called when the CBB docket selection dropdown changes.
    Uses the map storing FULL paths to files in the persistent CBB cache.
    """
    if not selected_docket or not docket_data_map:
        logging.warning("[CBB] Dropdown changed but missing selection or map. Cannot update preview.")
        return gr.update(value=None, label="Docket DataFrame Preview (Error: Missing data)")

    full_csv_path = docket_data_map.get(selected_docket)

    if not full_csv_path or not isinstance(full_csv_path, Path):
         logging.error(f"[CBB] Invalid or missing path stored for docket '{selected_docket}'. Map: {docket_data_map}")
         return gr.update(value=None, label=f"Docket DataFrame Preview (Error: Bad path for '{selected_docket}')")

    logging.info(f"[CBB] Dropdown changed to: {selected_docket}. Loading from cache: {full_csv_path}")

    try:
        if full_csv_path.exists() and full_csv_path.stat().st_size > 0:
            df = pl.read_csv(full_csv_path, try_parse_dates=True)
            preview_label = f"Preview: {selected_docket} ({full_csv_path.name})"
            logging.info(f"[CBB] Updated preview to show {selected_docket}")
            return gr.update(value=df, label=preview_label)
        elif not full_csv_path.exists():
             logging.warning(f"[CBB] Cached CSV file for selected docket '{selected_docket}' not found at: {full_csv_path}")
             return gr.update(value=None, label=f"Docket DataFrame Preview (Error: File not found for '{selected_docket}')")
        else: # File is empty
             logging.warning(f"[CBB] Cached CSV file for selected docket '{selected_docket}' is empty at: {full_csv_path}")
             return gr.update(value=None, label=f"Docket DataFrame Preview (Error: File empty for '{selected_docket}')")
    except Exception as e:
        logging.error(f"[CBB] Error reading cached CSV for {selected_docket} at {full_csv_path}: {e}", exc_info=True)
        return gr.update(value=None, label=f"Docket DataFrame Preview (Error reading {selected_docket})")

# --- Reset Function (CBB) ---
def reset_cbb_state(
    current_zip_path: Optional[str],
    preview_cache_dir: Optional[str]
) -> Tuple[
    None, str, None, None, # pdf_input, status_output, zip_output, zip_path_state
    Dict, Dict, Dict, None, # df_preview update, df_group update, docket_selector update, data_map_state clear
    None # preview_cache_dir_state clear
]: # Returns 9 items total
    """
    Clears the CBB tab UI elements, state, the persistent CBB preview cache directory,
    and the named temporary CBB zip file.
    """
    status = "[CBB] State reset."
    zip_removed_status = ""
    cache_removed_status = ""

    # --- Clean up named temporary zip file ---
    if current_zip_path:
        zip_path = Path(current_zip_path)
        logging.info(f"[CBB] Reset triggered. Attempting to remove temp zip file: {zip_path}")
        if zip_path.exists() and zip_path.is_file():
            try:
                zip_path.unlink()
                logging.info(f"[CBB] Successfully removed temporary zip file: {zip_path}")
                zip_removed_status = f" Removed temp file {zip_path.name}."
            except Exception as e:
                logging.error(f"[CBB] Error removing temporary zip file {zip_path}: {e}", exc_info=True)
                zip_removed_status = f" Error removing temp file {zip_path.name}: {e}."
        else:
            logging.warning(f"[CBB] Temporary zip file path found in state ({current_zip_path}), but file does not exist or is not a file.")
            zip_removed_status = f" Temp file {zip_path.name} not found."
    else:
        logging.info("[CBB] Reset triggered. No temp zip file path was stored.")

    # --- Clean up persistent preview cache directory ---
    if preview_cache_dir:
        cache_path = Path(preview_cache_dir)
        if cache_path.exists() and cache_path.is_dir():
            try:
                shutil.rmtree(cache_path)
                logging.info(f"[CBB] Successfully removed preview cache directory: {cache_path}")
                cache_removed_status = f" Removed preview cache {cache_path.name}."
            except Exception as e:
                logging.error(f"[CBB] Error removing preview cache directory {cache_path}: {e}", exc_info=True)
                cache_removed_status = f" Error removing preview cache {cache_path.name}: {e}."
        else:
            logging.warning(f"[CBB] Preview cache directory path found in state ({preview_cache_dir}), but directory does not exist or is not a directory.")
            cache_removed_status = f" Preview cache {cache_path.name} not found."
    else:
        logging.info("[CBB] No preview cache directory path found in state to remove.")

    # Combine status messages
    final_status = f"[CBB] State reset.{zip_removed_status}{cache_removed_status}"

    # Clear UI elements
    df_update = gr.update(value=None, label="Docket DataFrame Preview")
    group_update = gr.update(visible=False)
    dropdown_update = gr.update(choices=[], value=None, visible=False)
    map_clear = None
    cache_dir_clear = None

    # 9 items: pdf_input, status_output, zip_output, zip_path_state, df_preview, df_results_group, docket_selector_dd, data_map_state, preview_cache_dir_state
    return None, final_status, None, None, df_update, group_update, dropdown_update, map_clear, cache_dir_clear


# --- Gradio Interface Definition ---

with gr.Blocks(title="DHCR PDF Parser & Tools") as demo:
    gr.Markdown("# DHCR Tools")
    # --- Shared state for Vacancy Calculator Tab ---
    processed_units_data_state = gr.State(value={}) # Holds {unit_name: polars_df}

    with gr.Tabs():
        with gr.TabItem("DHCR Rent History Parser"): # --- TAB 1: PDF Parser (DHCR) ---
            gr.Markdown("## DHCR Rent History PDF Parser")
            gr.Markdown(
                "Upload a DHCR Rent History PDF file. The script will attempt to extract unit data using AI, "
                "optionally calculate vacancy allowances, generate images, and provide a downloadable zip file with the results."
                "\n**Note:** Processing can take several minutes depending on the PDF size and API response times."
                "\n**Requires a `GEMINI_API_KEY` environment variable or a `.env` file.**")

            # --- DHCR PDF Parser Layout START ---
            with gr.Row():
                with gr.Column(scale=1):
                    # Renamed components for clarity
                    dhcr_pdf_input = gr.File(label="Upload DHCR PDF", file_types=[".pdf"], type="filepath")
                    with gr.Row():
                        dhcr_images_checkbox = gr.Checkbox(label="Generate Page Images", value=True)
                        dhcr_vacancy_checkbox = gr.Checkbox(label="Calculate Vacancy Allowance (Alpha)", value=False)
                    dhcr_submit_button = gr.Button("Process DHCR PDF", variant="primary")
                    dhcr_reset_button = gr.Button("Reset DHCR Tab")
                with gr.Column(scale=2):
                    dhcr_status_output = gr.Textbox(label="Status / Logs", lines=8, interactive=False)
                    with gr.Group(visible=False) as dhcr_df_results_group:
                        dhcr_unit_selector_dd = gr.Dropdown(label="Select Unit to Preview", interactive=True, visible=False)
                        dhcr_df_preview = gr.DataFrame(label="DataFrame Preview", wrap=True)
                    dhcr_zip_output = gr.File(label="Download DHCR Results (.zip)")
                    # State components for DHCR Tab
                    dhcr_zip_path_state = gr.State(value=None)
                    dhcr_unit_data_map_state = gr.State(value={}) # Map: unit_name -> full_copied_csv_path
                    dhcr_preview_cache_dir_state = gr.State(value=None) # Path to persistent cache dir
            # --- DHCR PDF Parser Layout END ---

            # --- DHCR PDF Parser Event Handling ---
            dhcr_submit_button.click(
                fn=process_dhcr_pdf,
                inputs=[dhcr_pdf_input, dhcr_images_checkbox, dhcr_vacancy_checkbox],
                outputs=[
                    dhcr_zip_output, dhcr_status_output, dhcr_zip_path_state,
                    dhcr_df_preview, dhcr_df_results_group, dhcr_unit_selector_dd, dhcr_unit_data_map_state,
                    dhcr_preview_cache_dir_state, processed_units_data_state # Last one feeds Tab 2
                ]
            )

            dhcr_unit_selector_dd.change(
                fn=update_df_preview,
                inputs=[dhcr_unit_selector_dd, dhcr_unit_data_map_state],
                outputs=[dhcr_df_preview]
            )

            dhcr_reset_button.click(
                fn=reset_state,
                inputs=[dhcr_zip_path_state, dhcr_preview_cache_dir_state],
                outputs=[
                    dhcr_pdf_input, dhcr_status_output, dhcr_zip_output, dhcr_zip_path_state,
                    dhcr_df_preview, dhcr_df_results_group, dhcr_unit_selector_dd, dhcr_unit_data_map_state,
                    dhcr_preview_cache_dir_state, processed_units_data_state # Reset shared state too
                ]
            )
        # --- END TAB 1 (DHCR) ---

        # --- TAB 2: Vacancy Allowance Calculator ---
        with gr.TabItem("Vacancy Allowance Calculator"):
             if calculator_tab_available:
                  calculator_ui = create_calculator_tab(processed_units_data_state) # Pass shared state
             else:
                  gr.Markdown("## Vacancy Allowance Calculator")
                  gr.Markdown("**Error:** Failed to load the calculator tab module. Please check logs.")
        # --- END TAB 2 --- 

        # --- TAB 3: CBB Parser ---
        with gr.TabItem("DHCR CBB Reader"):
            gr.Markdown("## DHCR Cases By Building (CBB) PDF Parser")
            gr.Markdown(
                "Upload a DHCR Cases By Building (CBB) PDF file. The script will attempt to extract case data using AI, "
                "generate images of the pages, and provide a downloadable zip file containing CSVs (one per docket number) and images."
                "\n**Note:** Processing can take several minutes depending on the PDF size and API response times."
                "\n**Requires a `GEMINI_API_KEY` environment variable or a `.env` file.**")

            # --- CBB PDF Parser Layout START ---
            with gr.Row():
                with gr.Column(scale=1):
                    cbb_pdf_input = gr.File(label="Upload CBB PDF", file_types=[".pdf"], type="filepath")
                    cbb_images_checkbox = gr.Checkbox(label="Generate Page Images", value=True)
                    # No vacancy calc for CBB
                    cbb_submit_button = gr.Button("Process CBB PDF", variant="primary")
                    cbb_reset_button = gr.Button("Reset CBB Tab")
                with gr.Column(scale=2):
                    cbb_status_output = gr.Textbox(label="Status / Logs", lines=8, interactive=False)
                    # Group for CBB results area
                    with gr.Group(visible=False) as cbb_df_results_group:
                        cbb_docket_selector_dd = gr.Dropdown(label="Select Docket No. to Preview", interactive=True, visible=False)
                        cbb_df_preview = gr.DataFrame(label="Docket DataFrame Preview", wrap=True)

                    cbb_zip_output = gr.File(label="Download CBB Results (.zip)")
                    # State components for CBB Tab
                    cbb_zip_path_state = gr.State(value=None)
                    cbb_docket_data_map_state = gr.State(value={}) # Map: docket_no -> full_copied_csv_path
                    cbb_preview_cache_dir_state = gr.State(value=None) # Path to CBB persistent cache dir
            # --- CBB PDF Parser Layout END ---

            # --- CBB PDF Parser Event Handling ---
            cbb_submit_button.click(
                fn=process_cbb_pdf,
                inputs=[cbb_pdf_input, cbb_images_checkbox],
                outputs=[
                    cbb_zip_output, cbb_status_output, cbb_zip_path_state,
                    cbb_df_preview, cbb_df_results_group, cbb_docket_selector_dd, cbb_docket_data_map_state,
                    cbb_preview_cache_dir_state
                ]
            )

            cbb_docket_selector_dd.change(
                fn=update_cbb_df_preview,
                inputs=[cbb_docket_selector_dd, cbb_docket_data_map_state],
                outputs=[cbb_df_preview]
            )

            cbb_reset_button.click(
                fn=reset_cbb_state,
                inputs=[cbb_zip_path_state, cbb_preview_cache_dir_state],
                outputs=[
                    cbb_pdf_input, cbb_status_output, cbb_zip_output, cbb_zip_path_state,
                    cbb_df_preview, cbb_df_results_group, cbb_docket_selector_dd, cbb_docket_data_map_state,
                    cbb_preview_cache_dir_state
                ]
            )
        # --- END TAB 3 (CBB) ---


# --- Launch the App ---
if __name__ == "__main__":
    # Check for API key once at startup
    if not os.environ.get("GEMINI_API_KEY"):
         logging.warning("GEMINI_API_KEY not found in environment variables or .env file.")
         print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print("WARNING: GEMINI_API_KEY environment variable not set.")
         print("The application requires this key to function correctly.")
         print("Please set it in your environment or create a '.env' file")
         print("in the project root with: GEMINI_API_KEY=your_actual_key")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
         # Decide if you want to exit or continue with reduced functionality
         # exit(1)

    print("Launching Gradio App...")
    # Consider adding queue() if processing takes time, especially for PDF parsing
    # demo.queue()
    demo.launch() 