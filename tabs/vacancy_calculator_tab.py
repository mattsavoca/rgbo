import gradio as gr
from datetime import datetime, date
import logging
import polars as pl # Use polars for consistency if needed by imported functions
from typing import Tuple, Dict, Optional, List # Import Tuple for type hinting, Dict/Optional for footnote lookup, List for dropdown choices
from pathlib import Path # Ensure Path is imported
import tempfile # For temporary file creation for download

# Configure logging for this module
log = logging.getLogger(__name__)
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s] - %(message)s')

# --- Initialize RGBO data variable ---
rgb_data: Optional[pl.DataFrame] = None # Initialize here

# --- Import Logic Dependencies ---
try:
    # Use standard import relative to project root
    from scripts.calculate_vacancy_allowance import load_rgb_orders, get_order_details_for_date, DATE_RANGES, RGBO_CSV_PATH
    from scripts.calculate_vacancy_allowance import calculate_single_allowance_from_dict as backend_calculate_vacancy
    # Import the new function for batch processing
    from scripts.calculate_vacancy_allowance import add_vacancy_allowances_to_df 
    logging.info("Successfully imported calculation logic from scripts.calculate_vacancy_allowance")
    logic_available = True

    # --- Load RGBO data right after successful import ---
    log.info(f"Attempting to load RGBO data using load_rgb_orders from path hint: {RGBO_CSV_PATH}")
    rgb_data = load_rgb_orders() # <<< Load the data here
    if rgb_data is not None and not rgb_data.is_empty():
        log.info(f"Successfully loaded RGBO data. Shape: {rgb_data.shape}. Button should be enabled.")
    else:
        # This case will also disable the button, which is correct if loading fails.
        log.warning(f"load_rgb_orders completed but returned None or empty DataFrame. Path hint: {RGBO_CSV_PATH}. Button will be disabled.")
        rgb_data = None # Ensure it's None if loading failed

except ImportError as e:
    log.error(f"Fatal Error: Failed to import from calculate_vacancy_allowance: {e}. Calculator cannot function.", exc_info=True)
    rgb_data = None # Ensure it's None
    def get_order_details_for_date(*args, **kwargs): return None # type: ignore # Renamed dummy
    def backend_calculate_vacancy(*args, **kwargs): # type: ignore - For the interactive part
        log.error("backend_calculate_vacancy (interactive) is unavailable due to import error.")
        return "Error: Interactive calculation function unavailable due to import error."
    def add_vacancy_allowances_to_df(*args, **kwargs): # type: ignore - For the batch part
        log.error("add_vacancy_allowances_to_df (batch) is unavailable due to import error.")
        raise ImportError("add_vacancy_allowances_to_df is unavailable")
    DATE_RANGES = {}
    logic_available = False
except Exception as e:
    # This will catch errors during load_rgb_orders() as well
    log.error(f"Fatal Error: An unexpected error occurred during import or RGBO data loading: {e}. Calculator cannot function.", exc_info=True)
    rgb_data = None # Ensure it's None
    def get_order_details_for_date(*args, **kwargs): return None # type: ignore # Renamed dummy
    def backend_calculate_vacancy(*args, **kwargs): # type: ignore - For the interactive part
        log.error("backend_calculate_vacancy (interactive) is unavailable due to unexpected error.")
        return "Error: Interactive calculation function unavailable due to unexpected error."
    def add_vacancy_allowances_to_df(*args, **kwargs): # type: ignore - For the batch part
        log.error("add_vacancy_allowances_to_df (batch) is unavailable due to unexpected error.")
        raise ImportError("add_vacancy_allowances_to_df is unavailable")
    DATE_RANGES = {}
    logic_available = False

# --- Load Footnotes Data ---
# Define path relative to this script's location (in 'tabs')
SCRIPT_DIR = Path(__file__).parent.resolve() # 'tabs' directory
FOOTNOTES_CSV_PATH = SCRIPT_DIR.parent / "footnotes.csv" # Go up one level to root

footnotes_dict: Dict[str, str] = {}
try:
    # Use the absolute path
    log.info(f"Attempting to load footnotes from calculated path: {FOOTNOTES_CSV_PATH}")
    footnotes_df = pl.read_csv(FOOTNOTES_CSV_PATH).with_columns(pl.col("footnote_no").cast(pl.Utf8))
    footnotes_dict = dict(zip(footnotes_df["footnote_no"].to_list(), footnotes_df["note"].to_list()))
    log.info(f"Successfully loaded {len(footnotes_dict)} footnotes from {FOOTNOTES_CSV_PATH}.")
except FileNotFoundError:
    log.error(f"Footnotes file not found at {FOOTNOTES_CSV_PATH}. Footnote display will be unavailable.", exc_info=True)
except Exception as e:
    log.error(f"Error loading footnotes from {FOOTNOTES_CSV_PATH}: {e}", exc_info=True)


def get_footnote_text(footnote_num_str: Optional[str]) -> str:
    """Looks up footnote text by number string. Handles None or missing keys."""
    if footnote_num_str is None:
        return ""
    return footnotes_dict.get(footnote_num_str.strip(), f"Note {footnote_num_str} not found.")


def format_result(result):
    """Formats the numeric result as a percentage string or returns the string directly."""
    if isinstance(result, (int, float)):
        return f"{result * 100:.2f}%"
    return str(result) # Return strings like "Indeterminable..." as is

# --- Interactive Calculator Logic (Based on Flowchart Image) ---
def calculate_vacancy_allowance_interactive(
    apartment_status: str,
    is_new_tenant: str,
    term_length_str: str, # Use string from Radio
    lease_start_date: date, # Gradio Date component provides date object
    had_vacancy_allowance_in_prev_12_mo: str,
    previous_preferential_rent_has_value: str,
    tenant_tenure_years: float, # Gradio Number component provides float
    is_first_year: str,  # New parameter for RGBO Order #52 handling
    is_first_half: str   # New parameter for RGBO Order #53 handling
) -> Tuple[str, str, str]: # <--- Updated return type hint
    """
    Calculates vacancy allowance based on user inputs, following the flowchart image logic.
    Returns a tuple containing the formatted result string, a string explaining the calculation steps,
    and a string with relevant footnotes and guideline notes.
    """
    log.info("--- Starting Interactive Calculation ---")
    log.info(f"Inputs: Status='{apartment_status}', NewTenant='{is_new_tenant}', Term='{term_length_str}', "
             f"StartDate='{lease_start_date}', PrevVacAllow='{had_vacancy_allowance_in_prev_12_mo}', "
             f"PrevPrefRent='{previous_preferential_rent_has_value}', Tenure='{tenant_tenure_years}', "
             f"IsFirstYear='{is_first_year}', IsFirstHalf='{is_first_half}'")
    explanation_steps = ["Starting Calculation..."] # Initialize explanation log
    notes_details = ["Relevant Notes/Footnotes:"] # Initialize notes log

    if rgb_data is None or rgb_data.is_empty():
        log.error("Calculation aborted: RGBO data is not loaded.")
        error_msg = "Error: RGBO data failed to load. Cannot proceed." # Simplified error
        explanation_steps.append(error_msg)
        notes_details.append("N/A due to loading error.")
        return error_msg, "\n".join(explanation_steps), "\n".join(notes_details)
    explanation_steps.append("RGBO data loaded successfully.")

    if lease_start_date is None:
        log.warning("Calculation aborted: Lease Start Date is required.")
        error_msg = "Error: Lease Start Date is required."
        explanation_steps.append(error_msg)
        notes_details.append("N/A due to missing date.")
        return error_msg, "\n".join(explanation_steps), "\n".join(notes_details)
    # explanation_steps.append(f"Lease Start Date provided: {lease_start_date}") # Removed this intermediate step

    # --- Date Conversion Logic ---
    date_conversion_successful = False
    # Store original representation for logging
    original_input_date_repr = str(lease_start_date)
    lsd = lease_start_date  # Start with the original input
    
    try:
        # Handle various date input formats
        if isinstance(lsd, datetime):
            lsd = lsd.date()
            explanation_steps.append(f"Lease Start Date provided: {lsd} (from datetime)")
            date_conversion_successful = True
        elif isinstance(lsd, date):
            # Already a date object
            explanation_steps.append(f"Lease Start Date provided: {lsd}")
            date_conversion_successful = True
        elif isinstance(lsd, (float, int)):
            try:
                lsd = datetime.fromtimestamp(lsd).date()
                log.info(f"Converted numeric timestamp {original_input_date_repr} to date {lsd}")
                explanation_steps.append(f"Lease Start Date provided: {lsd} (from timestamp {original_input_date_repr})")
                date_conversion_successful = True
            except (OSError, ValueError, OverflowError) as e:
                log.error(f"Could not convert timestamp {original_input_date_repr} to date: {e}", exc_info=True)
                error_msg = f"Error: Invalid Lease Start Date timestamp: {original_input_date_repr}"
                explanation_steps.append(error_msg)
                notes_details.append("N/A due to invalid date.")
                return error_msg, "\n".join(explanation_steps), "\n".join(notes_details)
        elif isinstance(lsd, str):
            try:
                lsd = datetime.strptime(lsd, "%Y-%m-%d").date()
                explanation_steps.append(f"Lease Start Date provided: {lsd} (from string '{original_input_date_repr}')")
                date_conversion_successful = True
            except ValueError:
                log.error(f"Could not parse lease_start_date string: {original_input_date_repr}")
                error_msg = f"Error: Invalid Lease Start Date format: {original_input_date_repr}"
                explanation_steps.append(error_msg)
                notes_details.append("N/A due to invalid date.")
                return error_msg, "\n".join(explanation_steps), "\n".join(notes_details)
        else:
            log.error(f"Unsupported date type: {type(lease_start_date)} - Value: {original_input_date_repr}")
            error_msg = f"Error: Unsupported Lease Start Date format: {original_input_date_repr}"
            explanation_steps.append(error_msg)
            notes_details.append("N/A due to unsupported date format.")
            return error_msg, "\n".join(explanation_steps), "\n".join(notes_details)
    except Exception as e:
        log.error(f"Unexpected error processing lease_start_date: {e}", exc_info=True)
        error_msg = "Error: Failed to process Lease Start Date"
        explanation_steps.append(error_msg)
        notes_details.append("N/A due to date processing error.")
        return error_msg, "\n".join(explanation_steps), "\n".join(notes_details)
    
    # Check if conversion was successful
    if not date_conversion_successful: # Defensive check
        error_msg = "Error: Lease Start Date could not be processed."
        explanation_steps.append(error_msg)
        notes_details.append("N/A due to date processing error.")
        return error_msg, "\n".join(explanation_steps), "\n".join(notes_details)
    # --- End Date Conversion ---

    log.info(f"Fetching rates for date: {lsd}")
    explanation_steps.append(f"Fetching RGBO rates for date: {lsd}...")
    # --- Find Relevant RGBO Row ---
    try:
        # Filter the DataFrame to find the row where lsd falls between beginning_date and end_date
        # Convert date columns to date type first if they aren't already
        # if rgb_data["beginning_date"].dtype != pl.Date:
        #     rgb_data = rgb_data.with_columns(pl.col(["beginning_date", "end_date"]).str.strptime(pl.Date, "%Y-%m-%d", strict=False))

        # relevant_orders = rgb_data.filter(
        #     (pl.col("beginning_date") <= lsd) & (pl.col("end_date") >= lsd)
        # )
        # USE get_order_details_for_date instead of direct filtering here
        order_dict = get_order_details_for_date(rgb_data, lsd)

        if order_dict is None: # Check if get_order_details_for_date returned None
            log.warning(f"No RGBO rates found for date: {lsd} using get_order_details_for_date")
            error_msg = f"Indeterminable: No RGBO order found covering lease start date {lsd}"
            explanation_steps.append(error_msg)
            notes_details.append("N/A - No matching RGBO order found.")
            return error_msg, "\n".join(explanation_steps), "\n".join(notes_details)

        # No need to check len(relevant_orders) or get row [0] as order_dict is already the result or None
        # order_row = relevant_orders[0] # Get the first row as a Struct/Row object
        # order_dict = order_row.to_dicts()[0] # Convert the row to a dictionary for easier access

        log.info(f"Found matching RGBO Order: {order_dict.get('order_number', 'N/A')}")
        explanation_steps.append(f"Successfully retrieved RGBO Order {order_dict.get('order_number', 'N/A')} covering {lsd}.")

    except Exception as e:
        log.error(f"Error filtering RGBO data for date {lsd}: {e}", exc_info=True)
        error_msg = f"Error: Failed to process RGBO data for date {lsd}."
        explanation_steps.append(error_msg)
        notes_details.append("N/A due to data processing error.")
        return error_msg, "\n".join(explanation_steps), "\n".join(notes_details)
    # --- End Finding RGBO Row ---


    # Extract rates and notes from the dictionary
    one_yr_rate = order_dict.get('one_year_rate', 0.0)
    two_yr_rate = order_dict.get('two_year_rate', 0.0)
    vac_lease_rate = order_dict.get('vacancy_lease_rate') # Keep as None if missing initially
    one_yr_footnote = order_dict.get('one_year_footnote')
    two_yr_footnote = order_dict.get('two_year_footnote')
    vac_lease_footnotes = order_dict.get('vacancy_lease_footnotes')
    guideline_note = order_dict.get('guideline_note')
    guideline_note_footnotes = order_dict.get('guideline_note_footnotes')
    order_number = order_dict.get('order_number', 'Unknown') # Get order number for context
    
    # Extract additional rate data for RGBO Order #52 handling
    two_year_first_year_rate = order_dict.get('two_year_first_year_rate')
    two_year_first_year_rate_footnotes = order_dict.get('two_year_first_year_rate_footnotes')
    two_year_second_year_rate = order_dict.get('two_year_second_year_rate')
    two_year_second_year_rate_footnotes = order_dict.get('two_year_second_year_rate_footnotes')

    # Extract additional rate data for RGBO Order #53 handling
    one_year_first_half_rate = order_dict.get('one_year_first_half_rate')
    one_year_second_half_rate = order_dict.get('one_year_second_half_rate')

    # Handle potential None/Null values for rates
    one_yr_rate = 0.0 if one_yr_rate is None else float(one_yr_rate)
    two_yr_rate = 0.0 if two_yr_rate is None else float(two_yr_rate)
    # Vacancy lease rate requires special handling - 0.0 might be a valid rate
    has_specific_vac_rate = vac_lease_rate is not None
    vac_lease_rate = 0.0 if vac_lease_rate is None else float(vac_lease_rate)
    
    # Handle potential None/Null values for additional rates (Order #52)
    two_year_first_year_rate = 0.0 if two_year_first_year_rate is None else float(two_year_first_year_rate)
    two_year_second_year_rate = 0.0 if two_year_second_year_rate is None else float(two_year_second_year_rate)

    # Handle potential None/Null values for additional rates (Order #53)
    one_year_first_half_rate = 0.0 if one_year_first_half_rate is None else float(one_year_first_half_rate)
    one_year_second_half_rate = 0.0 if one_year_second_half_rate is None else float(one_year_second_half_rate)

    log.info(f"Rates from Order {order_number}: 1yr={one_yr_rate:.4f}, 2yr={two_yr_rate:.4f}, VacLease={vac_lease_rate:.4f} (Specific Rate Present: {has_specific_vac_rate})")
    
    # Additional logging for RGBO Order #52 special rates
    if order_number == "52":
        log.info(f"Order #52 Special Rates: 2yr-1st={two_year_first_year_rate:.4f}, 2yr-2nd={two_year_second_year_rate:.4f}")
        explanation_steps.append(f"Rates for Order {order_number}: 1-Year = {one_yr_rate*100:.2f}%, 2-Year = {two_yr_rate*100:.2f}%, First Year of 2-Year = {two_year_first_year_rate*100:.2f}%, Second Year of 2-Year = {two_year_second_year_rate*100:.2f}%, Specific Vacancy Lease Rate = {f'{vac_lease_rate*100:.2f}%' if has_specific_vac_rate else 'N/A'}")
    # Additional logging for RGBO Order #53 special rates
    elif order_number == "53":
        log.info(f"Order #53 Special Rates: 1yr-1stHalf={one_year_first_half_rate:.4f}, 1yr-2ndHalf={one_year_second_half_rate:.4f}")
        explanation_steps.append(f"Rates for Order {order_number}: 1-Year = {one_yr_rate*100:.2f}% (First 6 mo: {one_year_first_half_rate*100:.2f}%, Second 6 mo: {one_year_second_half_rate*100:.2f}%), 2-Year = {two_yr_rate*100:.2f}%, Specific Vacancy Lease Rate = {f'{vac_lease_rate*100:.2f}%' if has_specific_vac_rate else 'N/A'}")
    else:
        explanation_steps.append(f"Rates for Order {order_number}: 1-Year = {one_yr_rate*100:.2f}%, 2-Year = {two_yr_rate*100:.2f}%, Specific Vacancy Lease Rate = {f'{vac_lease_rate*100:.2f}%' if has_specific_vac_rate else 'N/A'}")

    # --- Compile Notes/Footnotes ---
    active_footnotes = set()
    notes_details = ["Relevant Notes/Footnotes:"] # Initialize notes log
    active_footnotes: Dict[str, list[str]] = {} # Footnote Num -> List of Sources
    notes_details = [] # Initialize notes log - Start empty

    def add_footnote(note_num_str: Optional[str], source_name: str):
        """Helper to add footnote numbers and their source to the dictionary."""
        if note_num_str is None: return
        for fn in str(note_num_str).split(','):
            fn_clean = fn.strip()
            if fn_clean:
                if fn_clean not in active_footnotes:
                    active_footnotes[fn_clean] = []
                # Avoid adding duplicate source names for the same footnote
                if source_name not in active_footnotes[fn_clean]:
                    active_footnotes[fn_clean].append(source_name)

    notes_details.append(f"**RGBO Order:** {order_number}")
    if guideline_note:
        notes_details.append(f"**Guideline Note:** {guideline_note}")
        add_footnote(guideline_note_footnotes, "Guideline Note")

    # Add footnotes based on which rate *might* be used in the logic below
    # Use the helper function to correctly associate the footnote with its source
    add_footnote(one_yr_footnote, "1-Year Rate")
    
    # Special handling for footnotes based on term length and RGBO order
    if term_length_str == "1":
        add_footnote(one_yr_footnote, "1-Year Rate")
    elif term_length_str == "2+":
        if order_number == "52":
            if is_first_year == "Yes":
                add_footnote(two_year_first_year_rate_footnotes, "2-Year Rate (Year 1)")
            else:  # is_first_year == "No"
                add_footnote(two_year_second_year_rate_footnotes, "2-Year Rate (Year 2)")
        else:  # other orders
            add_footnote(two_yr_footnote, "2-Year Rate")
    
    # Only add vac footnotes if a specific rate was present in the data
    if has_specific_vac_rate:
        add_footnote(vac_lease_footnotes, "Vacancy Lease Rate")

    if active_footnotes:
         notes_details.append('''\n **Applicable Footnotes:**''') # Add newline before header
         # Sort footnotes numerically for consistent order
         # Sort keys of the dictionary
         sorted_footnotes = sorted(active_footnotes.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
         for fn_num in sorted_footnotes:
             # Get sources from the dictionary value
             sources = ", ".join(active_footnotes[fn_num])
             note_text = get_footnote_text(fn_num)
             notes_details.append(f"- **[{fn_num}]** ({sources}): {note_text}") # Include sources in the output
    else:
        notes_details.append("No specific footnotes associated with this order's rates or guideline note.")

    # --- Flowchart Logic Implementation ---
    try:
        # Prepare unit_data dict for the backend function
        # Note: lsd is already processed into a Python date object by this point
        unit_data_for_backend = {
            'lease_start_date': lsd, # This is the processed Python date object
            'Apt Status': apartment_status,
            'is_new_tenant': (is_new_tenant == "Yes"), # Convert to bool for backend if it expects bool
            'term_length': 1 if term_length_str == "1" else 2, # Convert to int
            # For the interactive calculator, 'had_vacancy_allowance_in_prev_12_mo' is a direct input.
            # The backend calculate_vacancy_allowance_for_row will use this if previous_row_info is None.
            # So, we pass it directly if the backend function is designed to use it from the dict.
            # However, calculate_single_allowance_from_dict calls calculate_vacancy_allowance_for_row 
            # with previous_row_info=None, and calculate_vacancy_allowance_for_row then recalculates
            # had_vacancy_allowance_in_prev_12_mo_str based on that None. 
            # This means the direct input from the UI for this specific field might be tricky to reconcile
            # unless calculate_single_allowance_from_dict is smarter or expects it directly.
            # For now, let's assume the backend expects these fields directly in the dict passed to it.

            # The calculate_vacancy_allowance_for_row expects these keys in unit_data_current_row
            # The interactive UI provides these via separate Gradio components.
            'had_vacancy_allowance_in_prev_12_mo': (had_vacancy_allowance_in_prev_12_mo == "Yes"), # Pass as bool
            'previous_preferential_rent_has_value': (previous_preferential_rent_has_value == "Yes"), # Pass as bool
            'tenant_tenure_years': tenant_tenure_years, # Already float
            
            # These are needed by calculate_vacancy_allowance_for_row to determine certain conditions
            # but are effectively inputs to the interactive calculator.
            # The backend function will use these from the dict.
            'order_number_for_logic': order_dict.get('order_number', 'Unknown'), # Pass for Order 52/53 logic inside backend
            'is_order52_first_year_str_for_logic': "Yes" if (order_dict.get('order_number') == "52" and (1 if term_length_str == "1" else 2) >=2 and is_first_year == "Yes") else "No",
            'is_order53_first_half_str_for_logic': "First 6 Months" if (order_dict.get('order_number') == "53" and (1 if term_length_str == "1" else 2) == 1 and is_first_half == "First 6 Months") else "Second 6 Months",
        }

        # The interactive logic below was the actual calculation.
        # We now need to call the backend_calculate_vacancy function.
        # The original interactive logic is now *inside* calculate_vacancy_allowance_for_row.
        # So, we just need to call the backend function with the prepared dictionary.

        # Make sure all required inputs for `calculate_single_allowance_from_dict` (which calls `...for_row`)
        # are present in `unit_data_for_backend` or handled by `.get` within the backend.
        # `calculate_single_allowance_from_dict` primarily ensures `lease_start_date` is a date object.
        # The other fields are passed through to `calculate_vacancy_allowance_for_row`.

        result = backend_calculate_vacancy(unit_data_for_backend, rgb_data)
        explanation_steps.append(f"Called backend calculation. Result: {result}") # Simple explanation for now

        # The detailed explanation steps were part of the original interactive logic.
        # If we want those steps displayed, the backend function would need to return them too,
        # or we'd replicate a simplified version of that logic here just for explanation.
        # For now, the detailed step-by-step will be missing from the interactive output if only result is returned.

        # This was the original structure, assuming `result` is the final calculated value or error string.
        # is_pe_status = (apartment_status == "PE")
        # explanation_steps.append(f"1. Is Apartment Status 'PE'? {'Yes' if is_pe_status else 'No'}")
        # ... (rest of the original detailed logic which is now in the backend) ...

        formatted_result = format_result(result)
        explanation_steps.append(f"\nFinal Result: {formatted_result}")

        log.info(f"--- Calculation Complete. Result: {formatted_result} ---")
        explanation_log = "\n".join(explanation_steps)
        notes_log = "\n".join(notes_details) 

        return formatted_result, explanation_log, notes_log

    except KeyError as e:
         log.error(f"Missing date range key: {e}. DATE_RANGES might be incomplete.", exc_info=True)
         error_msg = f"Error: Internal configuration error (missing date range: {e})"
         explanation_steps.append(error_msg)
         notes_details.append("N/A due to configuration error.")
         return error_msg, "\n".join(explanation_steps), "\n".join(notes_details) # Return all three
    except Exception as e:
        log.error(f"An unexpected error occurred during calculation: {e}", exc_info=True)
        error_msg = f"Error: An unexpected calculation error occurred: {e}"
        explanation_steps.append(error_msg)
        notes_details.append("N/A due to calculation error.")
        return error_msg, "\n".join(explanation_steps), "\n".join(notes_details) # Return all three


# --- Gradio UI Definition ---
def create_calculator_tab(processed_data_state: gr.State): # <<< Accept shared state
    """Creates the Gradio Blocks UI for the Vacancy Allowance Calculator tab.

    Args:
        processed_data_state: A Gradio State object holding the dictionary
                              of {unit_name: polars_dataframe} from the first tab.
    """
    with gr.Blocks() as calculator_tab:
        gr.Markdown("## Vacancy Allowance Calculator")
        gr.Markdown("Enter the details below as per the flowchart to calculate the allowance.")

        if rgb_data is None or rgb_data.is_empty():
             gr.Markdown("**Error: RGBO data failed to load. Calculator is non-functional.** Check application logs.", elem_id="error-message")

        # --- ADDED: Preview Section ---
        with gr.Accordion("Preview Extracted Unit Data", open=False):
            calculator_unit_selector_dd = gr.Dropdown(
                label="Select Unit Data to Preview",
                choices=[],
                interactive=True,
                info="Select a unit processed in the PDF Parser tab to view its data."
            )
            calculator_df_preview = gr.DataFrame(
                label="Selected Unit Data Preview",
                interactive=False
            )
        # --- END Preview Section ---

        # --- ADDED: Batch Processing Section for Selected Unit CSV ---
        with gr.Group():
            gr.Markdown("### Process Full Unit CSV for Vacancy Allowances")
            gr.Markdown("Select a unit from the dropdown above. Then click the button below to calculate vacancy allowances for all rows in its CSV and add it as a new column ('x_vacancy_allowance').")
            process_selected_csv_button = gr.Button(
                "Calculate & Add Allowances to Selected Unit CSV",
                variant="secondary",
                interactive=(logic_available and rgb_data is not None and not rgb_data.is_empty())
            )
            processed_csv_status_output = gr.Textbox(
                label="Batch Processing Status", 
                interactive=False, 
                lines=3
            )
            calculator_df_preview_with_allowances = gr.DataFrame(
                label="Preview of CSV with Added 'x_vacancy_allowance' Column",
                interactive=False
            )
            download_augmented_csv_button = gr.File(
                label="Download Augmented CSV with Allowances",
                interactive=True # Will be made interactive once a file is ready
            )
        # --- END Batch Processing Section ---

        # --- ADDED: Combine to XLSX Section ---
        with gr.Group():
            gr.Markdown("### Combine All Processed Units to XLSX")
            gr.Markdown(
                "Click the button below to process all units currently loaded from the PDF Parser tab. "
                "This will apply vacancy allowance calculations to each unit, transform columns "
                "as per the specified XLSX format, and generate a single XLSX file with each unit as a sheet."
            )
            combine_to_xlsx_button = gr.Button(
                "Combine All Units to XLSX",
                variant="secondary",
                # Interactivity will depend on logic_available, rgb_data, and if processed_data_state has items
                interactive=(logic_available and rgb_data is not None and not rgb_data.is_empty())
            )
            xlsx_status_output = gr.Textbox(
                label="XLSX Generation Status",
                interactive=False,
                lines=3
            )
            download_xlsx_file = gr.File(
                label="Download Combined XLSX File",
                interactive=True # Initial state, will be updated by button click
            )
        # --- END Combine to XLSX Section ---

        with gr.Row():
            with gr.Column(scale=1):
                apt_status_input = gr.Radio(
                    label="Apartment Status",
                    choices=["PE", "Other"],
                    value="Other",
                    info="Is the status 'PE'?"
                )
                new_tenant_input = gr.Radio(
                    label="New Tenant",
                    choices=["Yes", "No"],
                    value="Yes",
                    info="Is this a new tenant?"
                )
                term_length_input = gr.Radio(
                    label="Term Length",
                    choices=["1", "2+"],
                    value="1",
                    info="Select the lease term length."
                )
                lease_start_input = gr.DateTime( # Use gr.DateTime instead of gr.Date
                    label="Lease Start Date",
                    info="The start date of the lease in question.",
                    include_time=False # Set to False to only show the date picker
                )
                # Input for RGBO Order #52 handling (2-year), hidden by default
                is_first_year_input = gr.Radio(
                    label="Is this the first year of the 2-year lease?",
                    choices=["Yes", "No"],
                    value="Yes",
                    visible=False,  # Hidden by default
                    info="Visible only for 2-year leases during RGBO Order #52 period (10/1/2020-9/30/2021)."
                )
                # New input for RGBO Order #53 handling (1-year), hidden by default
                is_first_half_input = gr.Radio(
                    label="Is this the first or second 6 months of the 1-year lease?",
                    choices=["First 6 Months", "Second 6 Months"],
                    value="First 6 Months",
                    visible=False,  # Hidden by default
                    info="Visible only for 1-year leases during RGBO Order #53 period (10/1/2021-9/30/2022)."
                )

            with gr.Column(scale=1):
                prev_vac_allow_input = gr.Radio(
                    label="Vacancy Allowance in previous 12 mo?",
                    choices=["Yes", "No"],
                    value="No",
                    # info="Relevant for specific date ranges and terms."
                )
                prev_pref_rent_input = gr.Radio(
                    label="Previous-year Preferential Rent has value?",
                    choices=["Yes", "No"],
                    value="No",
                     # info="Relevant if vacancy allowance taken previously."
                )
                tenant_tenure_input = gr.Number(
                    label="Tenant Tenure (Years)",
                    value=0,
                    minimum=0,
                    info="Enter the number of years the tenant has resided (if applicable)."
                )

        calculate_button = gr.Button("Calculate Allowance", variant="primary", interactive=(rgb_data is not None and not rgb_data.is_empty()))
        result_output = gr.Textbox(label="Calculated Vacancy Allowance", interactive=False)
        # Add the new Textbox for the explanation log
        logic_output = gr.Textbox(label="Calculation Logic", interactive=False, lines=10)
        # Add a new Markdown component for notes/footnotes
        notes_output = gr.Markdown(label="Relevant Guideline Notes & Footnotes", value="Notes will appear here after calculation.")
        
        # --- Import for XLSX generation ---
        try:
            from scripts.units_to_xls import generate_xlsx_from_units_data
            xlsx_logic_available = True
            log.info("Successfully imported generate_xlsx_from_units_data from scripts.units_to_xls")
        except ImportError as e:
            log.error(f"Failed to import XLSX generation logic: {e}", exc_info=True)
            xlsx_logic_available = False
            def generate_xlsx_from_units_data(*args, **kwargs):
                raise ImportError("generate_xlsx_from_units_data is unavailable due to import error.")
        # --- End Import for XLSX ---

        # --- ADDED: Function and Event Handler for Batch CSV Processing ---
        def process_selected_unit_csv_for_tab(
            selected_unit_name: Optional[str], 
            current_processed_data: Dict[str, pl.DataFrame],
            # rgb_data is accessed from the global scope within this tab's context
        ) -> Tuple[str, Optional[pl.DataFrame], Optional[str], gr.update]: # Added gr.update for visibility
            visibility_update = gr.update(visible=False) # Default to not visible
            if not logic_available or rgb_data is None or rgb_data.is_empty():
                status = "Error: Core calculation logic or RGBO data is not available. Cannot process."
                log.error(status)
                return status, None, None, visibility_update

            if not selected_unit_name:
                status = "Please select a unit from the 'Preview Extracted Unit Data' dropdown first."
                log.warning(status)
                return status, None, None, visibility_update

            if not current_processed_data or selected_unit_name not in current_processed_data:
                status = f"Error: Data for selected unit '{selected_unit_name}' not found in the processed data cache."
                log.error(status)
                return status, None, None, visibility_update

            unit_df_to_process = current_processed_data[selected_unit_name]
            if not isinstance(unit_df_to_process, pl.DataFrame) or unit_df_to_process.is_empty():
                status = f"Error: DataFrame for unit '{selected_unit_name}' is invalid or empty."
                log.error(status)
                return status, None, None, visibility_update

            status = f"Processing CSV for unit: {selected_unit_name}...\nInput shape: {unit_df_to_process.shape}"
            log.info(status)

            try:
                df_with_allowances = add_vacancy_allowances_to_df(unit_df_to_process, rgb_data)
                
                status += f"\nProcessing complete. Output shape: {df_with_allowances.shape}. Contains 'x_vacancy_allowance' column."
                log.info(f"Successfully processed CSV for {selected_unit_name}. Output shape: {df_with_allowances.shape}")

                # Prepare DataFrame for CSV writing: cast Object columns (like x_vacancy_allowance) to String
                df_to_write = df_with_allowances.clone() # Work on a copy for writing
                for col_name in df_to_write.columns:
                    if df_to_write[col_name].dtype == pl.Object:
                        log.info(f"Casting column '{col_name}' from Object to String using map_elements before CSV writing.")
                        df_to_write = df_to_write.with_columns(pl.col(col_name).map_elements(str, return_dtype=pl.String))
                    # Optional: Cast other types like Date/Datetime to string if default format is not desired
                    # For now, Polars default string conversion for dates should be fine.

                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix=f"{selected_unit_name}_with_allowances_") as tmp_file:
                    df_to_write.write_csv(tmp_file.name)
                    download_path = tmp_file.name
                
                status += f"\nAugmented CSV ready for download."
                log.info(f"Augmented CSV for {selected_unit_name} saved to temp file: {download_path}")
                visibility_update = gr.update(visible=True) # Make button visible on success
                return status, df_with_allowances, download_path, visibility_update

            except ImportError as imp_err:
                error_msg = f"Error during processing for {selected_unit_name}: Could not use processing function ({imp_err}). Ensure imports are correct."
                log.error(error_msg, exc_info=True)
                return error_msg, None, None, visibility_update
            except Exception as e:
                error_msg = f"An unexpected error occurred while processing CSV for {selected_unit_name}: {e}"
                log.error(error_msg, exc_info=True)
                return error_msg, None, None, visibility_update

        process_selected_csv_button.click(
            fn=process_selected_unit_csv_for_tab,
            inputs=[
                calculator_unit_selector_dd, 
                processed_data_state         
            ],
            outputs=[
                processed_csv_status_output,
                calculator_df_preview_with_allowances,
                download_augmented_csv_button, # This will receive the path or None
                download_augmented_csv_button  # This will receive the visibility update
            ]
        )
        # --- END Batch CSV Processing --- 

        # --- ADDED: Function and Event Handler for Combine to XLSX ---
        def handle_combine_to_xlsx(
            current_processed_data: Dict[str, pl.DataFrame]
            # rgb_data is accessed from the global scope (rgb_data in this file)
        ) -> Tuple[str, Optional[str], gr.update]: # Status, filepath, visibility update for download
            visibility_update = gr.update(value=None, visible=False) # Default to hidden and no file

            if not xlsx_logic_available:
                status = "Error: XLSX generation logic is not available. Cannot process."
                log.error(status)
                return status, None, visibility_update
            
            if not logic_available or rgb_data is None or rgb_data.is_empty():
                status = "Error: Core calculation logic or RGBO data is not available. Cannot process for XLSX."
                log.error(status)
                return status, None, visibility_update

            if not current_processed_data:
                status = "No unit data loaded from the PDF Parser tab. Please process a PDF first."
                log.warning(status)
                return status, None, visibility_update

            status = f"Starting XLSX generation for {len(current_processed_data)} units..."
            log.info(status)

            try:
                xlsx_filepath = generate_xlsx_from_units_data(current_processed_data, rgb_data)

                if xlsx_filepath:
                    status += f"\nSuccessfully generated XLSX file: {Path(xlsx_filepath).name}"
                    log.info(f"XLSX file ready for download: {xlsx_filepath}")
                    visibility_update = gr.update(value=xlsx_filepath, visible=True)
                    return status, xlsx_filepath, visibility_update
                else:
                    status += "\nError: XLSX generation failed. Check logs for details."
                    log.error("XLSX generation returned no filepath.")
                    return status, None, visibility_update

            except ImportError as imp_err: # Should be caught by xlsx_logic_available check mostly
                error_msg = f"Error during XLSX generation: Could not use processing function ({imp_err})."
                log.error(error_msg, exc_info=True)
                return error_msg, None, visibility_update
            except Exception as e:
                error_msg = f"An unexpected error occurred during XLSX generation: {e}"
                log.error(error_msg, exc_info=True)
                return error_msg, None, visibility_update

        combine_to_xlsx_button.click(
            fn=handle_combine_to_xlsx,
            inputs=[processed_data_state], # Pass the shared state
            outputs=[
                xlsx_status_output,
                download_xlsx_file,      # For the file path
                download_xlsx_file       # For visibility update
            ]
        )
        # --- END Combine to XLSX ---

        # Function to update tenant tenure input based on new tenant status
        def update_tenure_interactivity(new_tenant_status):
            if new_tenant_status == "Yes":
                return gr.update(value=0, interactive=False)
            else:
                return gr.update(interactive=True)

        # Function to determine visibility of conditional inputs based on term length and date
        def update_conditional_visibility(term, start_date):
            show_order_52_q = False
            show_order_53_q = False
            try:
                # Process the date (convert from various types)
                if start_date is None:
                    return gr.update(visible=False), gr.update(visible=False)
                
                processed_date = None
                if isinstance(start_date, str):
                    try: processed_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                    except ValueError: log.error(f"Invalid date string format: {start_date}")
                elif isinstance(start_date, datetime): processed_date = start_date.date()
                elif isinstance(start_date, (float, int)):
                    try: processed_date = datetime.fromtimestamp(start_date).date()
                    except (ValueError, OSError, OverflowError) as e: log.error(f"Cannot convert timestamp {start_date} to date: {e}")
                elif isinstance(start_date, date): processed_date = start_date
                else: log.error(f"Could not convert {start_date} (type: {type(start_date)}) to date")

                if processed_date:
                    # Check Order #52 (2-year term, 10/1/2020 - 9/30/2021)
                    if term == "2+":
                        order_52_start = date(2020, 10, 1)
                        order_52_end = date(2021, 9, 30)
                        if order_52_start <= processed_date <= order_52_end:
                            log.info(f"Showing Order #52 question: Lease date {processed_date} is within period")
                            show_order_52_q = True
                    
                    # Check Order #53 (1-year term, 10/1/2021 - 9/30/2022)
                    elif term == "1":
                        order_53_start = date(2021, 10, 1)
                        order_53_end = date(2022, 9, 30)
                        if order_53_start <= processed_date <= order_53_end:
                            log.info(f"Showing Order #53 question: Lease date {processed_date} is within period")
                            show_order_53_q = True
                            
            except Exception as e:
                log.error(f"Error in update_conditional_visibility: {e}", exc_info=True)
            
            # Return updates for both conditional inputs
            return gr.update(visible=show_order_52_q), gr.update(visible=show_order_53_q)
        
        # Event handlers for conditionally showing the relevant questions
        term_length_input.change(
            fn=update_conditional_visibility,
            inputs=[term_length_input, lease_start_input],
            outputs=[is_first_year_input, is_first_half_input] # Update both
        )
        
        lease_start_input.change(
            fn=update_conditional_visibility,
            inputs=[term_length_input, lease_start_input],
            outputs=[is_first_year_input, is_first_half_input] # Update both
        )

        # Event handler to update tenure input interactivity
        new_tenant_input.change(
            fn=update_tenure_interactivity,
            inputs=[new_tenant_input],
            outputs=[tenant_tenure_input]
        )

        # Event Handler for calculation
        calculate_button.click(
            fn=calculate_vacancy_allowance_interactive,
            inputs=[
                apt_status_input,
                new_tenant_input,
                term_length_input,
                lease_start_input,
                prev_vac_allow_input,
                prev_pref_rent_input,
                tenant_tenure_input,
                is_first_year_input,
                is_first_half_input  # Add the new input here
            ],
            outputs=[result_output, logic_output, notes_output]
        )

        # --- ADDED: Event Handlers for Preview Section ---

        # Function to update the dropdown choices when the shared state changes
        def update_calculator_unit_dropdown(processed_data: Dict[str, pl.DataFrame]) -> gr.Dropdown:
            """Updates the choices in the unit selection dropdown based on the processed data.
            Args:
                processed_data: The dictionary {unit_name: dataframe} from the shared state.
            Returns:
                An updated Gradio Dropdown component.
            """
            sorted_unit_names = sorted(list(processed_data.keys())) if processed_data else []
            log.info(f"Updating calculator tab dropdown with units: {sorted_unit_names}")
            # Select the first unit name as the default value if units exist, otherwise None
            new_value = sorted_unit_names[0] if sorted_unit_names else None
            log.info(f"Setting dropdown value to: {new_value}")
            # Keep existing value if it's still valid, otherwise reset
            # current_value = calculator_unit_selector_dd.value # Can't access component value directly here
            # new_value = current_value if current_value in unit_names else None
            # return gr.update(choices=sorted(unit_names), value=None, interactive=bool(unit_names)) # Old return
            return gr.update(choices=sorted_unit_names, value=new_value, interactive=bool(sorted_unit_names)) # New return with default value

        # Function to update the DataFrame preview when a unit is selected
        def update_calculator_df_preview(selected_unit: str, processed_data: Dict[str, pl.DataFrame]) -> gr.DataFrame:
            """Updates the DataFrame component to show the data for the selected unit.
            Args:
                selected_unit: The unit name selected in the dropdown.
                processed_data: The dictionary {unit_name: dataframe} from the shared state.
            Returns:
                An updated Gradio DataFrame component.
            """
            if selected_unit and processed_data and selected_unit in processed_data:
                log.info(f"Displaying DataFrame for selected unit in calculator tab: {selected_unit}")
                return gr.update(value=processed_data[selected_unit])
            else:
                log.info(f"Clearing calculator DataFrame preview (selected: {selected_unit}, data present: {bool(processed_data)})")
                return gr.update(value=None)

        # Trigger dropdown update when the shared state changes
        processed_data_state.change(
            fn=update_calculator_unit_dropdown,
            inputs=[processed_data_state],
            outputs=[calculator_unit_selector_dd]
        )

        # Trigger DataFrame update when the dropdown selection changes
        calculator_unit_selector_dd.change(
            fn=update_calculator_df_preview,
            inputs=[calculator_unit_selector_dd, processed_data_state],
            outputs=[calculator_df_preview]
        )
        # --- END Event Handlers for Preview Section ---

    # Return the Blocks object so it can be rendered in the main app
    return calculator_tab

# Example of running just this tab for testing (optional)
if __name__ == '__main__':
    print("Launching Vacancy Calculator Tab standalone for testing...")
    # Create a dummy state for testing if needed
    class DummyState:
        def __init__(self, value):
            self.value = value
        def change(self, fn, inputs, outputs):
            pass # No-op for simple testing
    
    dummy_processed_data = {}
    # To test the batch processor, you might want to populate dummy_processed_data
    # with a sample Polars DataFrame, e.g.:
    # sample_df_data = {
    #     'Tenant Name': ['Tenant A', 'Tenant A', 'Tenant B'],
    #     'Lease Began': [date(2020,1,1), date(2021,1,1), date(2020,6,1)],
    #     'Lease Ends': [date(2020,12,31), date(2021,12,31), date(2021,5,30)],
    #     'Actual Rent Paid': [1000, 1050, 1200],
    #     'Legal Reg Rent': [1000, 1050, 1200],
    #     'Apt Status': ['Other','Other','Other']
    # }
    # try:
    #     sample_df = pl.DataFrame(sample_df_data)
    #     dummy_processed_data = {"Unit1_Test": sample_df}
    #     print("Created dummy DataFrame for testing.")
    # except Exception as e:
    #     print(f"Could not create dummy polars DataFrame: {e}")

    dummy_state = DummyState(dummy_processed_data) 

    if rgb_data is None:
         print("\nWARNING: RGBO data failed to load. The calculator will show an error message and some parts may be disabled.\n")
    interface = create_calculator_tab(dummy_state) # Pass dummy state
    interface.launch() 