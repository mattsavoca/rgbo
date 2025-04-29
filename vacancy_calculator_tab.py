import gradio as gr
from datetime import datetime, date
import logging
import polars as pl # Use polars for consistency if needed by imported functions
from typing import Tuple, Dict, Optional # Import Tuple for type hinting, Dict/Optional for footnote lookup

# Configure logging for this module
log = logging.getLogger(__name__)
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s] - %(message)s')

# --- Import Core Logic ---
try:
    # Assuming rgbo.csv is in the root and calculate_vacancy_allowance.py is importable
    from calculate_vacancy_allowance import load_rgb_orders, get_rates_for_date, DATE_RANGES, RGBO_CSV_PATH
    log.info("Successfully imported functions from calculate_vacancy_allowance.")
    # Load RGBO data once when the module is loaded
    rgb_data = load_rgb_orders(RGBO_CSV_PATH)
    log.info(f"RGBO data loaded successfully for the calculator tab. Shape: {rgb_data.shape}")
    # Ensure date columns are converted immediately after loading
    if rgb_data is not None and not rgb_data.is_empty():
        if rgb_data["beginning_date"].dtype != pl.Date:
             try:
                log.info("Converting date columns in global RGBO data to Date type...")
                rgb_data = rgb_data.with_columns(pl.col(["beginning_date", "end_date"]).str.strptime(pl.Date, "%Y-%m-%d", strict=False))
                log.info("Date columns converted successfully.")
             except Exception as e:
                 log.error(f"Failed to convert date columns in RGBO data: {e}", exc_info=True)
                 rgb_data = None # Mark data as unusable if conversion fails
except FileNotFoundError as e:
    log.error(f"Fatal Error: RGBO CSV file not found at expected path ({RGBO_CSV_PATH}). Calculator cannot function. Error: {e}", exc_info=True)
    rgb_data = None # Indicate data loading failure
    # Define dummy functions to prevent NameErrors later if import failed partially
    def get_rates_for_date(*args, **kwargs): return None
    DATE_RANGES = {}
except ImportError as e:
    log.error(f"Fatal Error: Failed to import from calculate_vacancy_allowance: {e}. Calculator cannot function.", exc_info=True)
    rgb_data = None
    def get_rates_for_date(*args, **kwargs): return None
    DATE_RANGES = {}
except Exception as e:
    log.error(f"Fatal Error: An unexpected error occurred during import or RGBO data loading: {e}. Calculator cannot function.", exc_info=True)
    rgb_data = None
    def get_rates_for_date(*args, **kwargs): return None
    DATE_RANGES = {}


# --- Load Footnotes Data ---
FOOTNOTES_CSV_PATH = "footnotes.csv"
footnotes_dict: Dict[str, str] = {}
try:
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
    tenant_tenure_years: float # Gradio Number component provides float
) -> Tuple[str, str, str]: # <--- Updated return type hint
    """
    Calculates vacancy allowance based on user inputs, following the flowchart image logic.
    Returns a tuple containing the formatted result string, a string explaining the calculation steps,
    and a string with relevant footnotes and guideline notes.
    """
    log.info("--- Starting Interactive Calculation ---")
    log.info(f"Inputs: Status='{apartment_status}', NewTenant='{is_new_tenant}', Term='{term_length_str}', "
             f"StartDate='{lease_start_date}', PrevVacAllow='{had_vacancy_allowance_in_prev_12_mo}', "
             f"PrevPrefRent='{previous_preferential_rent_has_value}', Tenure='{tenant_tenure_years}'")
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

    # Ensure lease_start_date is a date object
    lsd = lease_start_date
    original_input_date_repr = str(lease_start_date) # Store original representation for logging

    # --- Date Conversion Logic ---
    date_conversion_successful = False
    if isinstance(lsd, datetime):
        lsd = lsd.date()
        explanation_steps.append(f"Lease Start Date provided: {lsd} (from datetime)")
        date_conversion_successful = True
    elif isinstance(lsd, float):
        try:
            lsd = datetime.fromtimestamp(lsd).date()
            log.info(f"Converted float timestamp {original_input_date_repr} to date {lsd}")
            explanation_steps.append(f"Lease Start Date provided: {lsd} (from timestamp {original_input_date_repr})")
            date_conversion_successful = True
        except (OSError, ValueError) as e:
             log.error(f"Could not convert float timestamp {original_input_date_repr} to date: {e}", exc_info=True)
             error_msg = f"Error: Invalid Lease Start Date timestamp: {original_input_date_repr}"
             explanation_steps.append(error_msg)
             notes_details.append("N/A due to invalid date.")
             return error_msg, "\n".join(explanation_steps), "\n".join(notes_details)
    elif not isinstance(lsd, date):
        try:
             lsd = datetime.strptime(str(lsd), "%Y-%m-%d").date()
             explanation_steps.append(f"Lease Start Date provided: {lsd} (from string '{original_input_date_repr}')")
             date_conversion_successful = True
        except (ValueError, TypeError):
            log.error(f"Could not parse lease_start_date: {original_input_date_repr} (Type: {type(lease_start_date)})")
            error_msg = f"Error: Invalid Lease Start Date format: {original_input_date_repr}"
            explanation_steps.append(error_msg)
            notes_details.append("N/A due to invalid date.")
            return error_msg, "\n".join(explanation_steps), "\n".join(notes_details)
    else: # Already a date object
        explanation_steps.append(f"Lease Start Date provided: {lsd}")
        date_conversion_successful = True

    # Check if conversion was successful before proceeding (should always be true if no error returned)
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

        relevant_orders = rgb_data.filter(
            (pl.col("beginning_date") <= lsd) & (pl.col("end_date") >= lsd)
        )

        if relevant_orders.is_empty():
            log.warning(f"No RGBO rates found for date: {lsd}")
            error_msg = f"Indeterminable: No RGBO order found covering lease start date {lsd}"
            explanation_steps.append(error_msg)
            notes_details.append("N/A - No matching RGBO order found.")
            return error_msg, "\n".join(explanation_steps), "\n".join(notes_details)

        # Assuming only one order applies per date. If multiple, log warning and take the first.
        if len(relevant_orders) > 1:
            log.warning(f"Multiple RGBO orders ({relevant_orders['order_number'].to_list()}) found for {lsd}. Using the first one: {relevant_orders[0, 'order_number']}")
        
        order_row = relevant_orders[0] # Get the first row as a Struct/Row object
        order_dict = order_row.to_dicts()[0] # Convert the row to a dictionary for easier access

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


    # Handle potential None/Null values for rates
    one_yr_rate = 0.0 if one_yr_rate is None else float(one_yr_rate)
    two_yr_rate = 0.0 if two_yr_rate is None else float(two_yr_rate)
    # Vacancy lease rate requires special handling - 0.0 might be a valid rate
    has_specific_vac_rate = vac_lease_rate is not None
    vac_lease_rate = 0.0 if vac_lease_rate is None else float(vac_lease_rate)

    log.info(f"Rates from Order {order_number}: 1yr={one_yr_rate:.4f}, 2yr={two_yr_rate:.4f}, VacLease={vac_lease_rate:.4f} (Specific Rate Present: {has_specific_vac_rate})")
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
    add_footnote(two_yr_footnote, "2-Year Rate")
    # Only add vac footnotes if a specific rate was present in the data
    if has_specific_vac_rate:
        add_footnote(vac_lease_footnotes, "Vacancy Lease Rate")

    # Remove the original (now redundant) logic that called add_footnote again.
    # The logic above now handles adding the footnotes with their sources.

    if active_footnotes:
         notes_details.append("\\n**Applicable Footnotes:**") # Add newline before header
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
        is_pe_status = (apartment_status == "PE")
        explanation_steps.append(f"1. Is Apartment Status 'PE'? {'Yes' if is_pe_status else 'No'}")

        if is_pe_status:
            log.info("Path: Apartment Status == PE")
            result = one_yr_rate
            explanation_steps.append(f"--> Result: 1-Year Rate ({one_yr_rate*100:.2f}%)")
        else: # Not PE
            log.info("Path: Apartment Status != PE")
            is_new = (is_new_tenant == "Yes")
            explanation_steps.append(f"2. Is this a New Tenant? {'Yes' if is_new else 'No'}")
            if not is_new:
                log.info("Path: Not New Tenant")
                result = 0.0
                explanation_steps.append("--> Result: 0.00%")
            else: # New Tenant == Yes
                log.info("Path: New Tenant == Yes")
                term_length = 1 if term_length_str == "1" else 2
                explanation_steps.append(f"3. Lease Term Length: {term_length_str}")

                if term_length == 1:
                    log.info("Path: Term Length == 1")
                    explanation_steps.append(f"4. Checking Lease Start Date ({lsd}) for 1-Year Term...")
                    # Check Lease Start Date (Term 1)
                    if DATE_RANGES["range_83_97"][0] <= lsd <= DATE_RANGES["range_83_97"][1]:
                        log.info("Path: Lease Date 83-97")
                        explanation_steps.append(f"--> Date falls within {DATE_RANGES['range_83_97'][0]} to {DATE_RANGES['range_83_97'][1]}.")
                        result = one_yr_rate + vac_lease_rate
                        explanation_steps.append(f"--> Calculation: 1-Year Rate ({one_yr_rate*100:.2f}%) + Vacancy Lease Rate ({vac_lease_rate*100:.2f}%) = {result*100:.2f}%")
                    elif DATE_RANGES["range_97_11"][0] <= lsd <= DATE_RANGES["range_97_11"][1]:
                        log.info("Path: Lease Date 97-11")
                        explanation_steps.append(f"--> Date falls within {DATE_RANGES['range_97_11'][0]} to {DATE_RANGES['range_97_11'][1]}.")
                        result = 0.20 - (two_yr_rate - one_yr_rate)
                        explanation_steps.append(f"--> Calculation: 20.00% - (2-Year Rate ({two_yr_rate*100:.2f}%) - 1-Year Rate ({one_yr_rate*100:.2f}%)) = {result*100:.2f}%")
                    elif DATE_RANGES["range_11_15"][0] <= lsd <= DATE_RANGES["range_11_15"][1]:
                         log.info("Path: Lease Date 11-15")
                         explanation_steps.append(f"--> Date falls within {DATE_RANGES['range_11_15'][0]} to {DATE_RANGES['range_11_15'][1]}.")
                         result = 0.0
                         explanation_steps.append("--> Result: 0.00%")
                    elif DATE_RANGES["range_15_19"][0] <= lsd <= DATE_RANGES["range_15_19"][1]:
                        log.info("Path: Lease Date 15-19")
                        explanation_steps.append(f"--> Date falls within {DATE_RANGES['range_15_19'][0]} to {DATE_RANGES['range_15_19'][1]}.")
                        had_prev_vac = (had_vacancy_allowance_in_prev_12_mo == "Yes")
                        explanation_steps.append(f"5. Vacancy Allowance taken in prior 12 mo? {'Yes' if had_prev_vac else 'No'}")
                        if had_prev_vac:
                            log.info("Path: Had Vacancy Allowance Prev 12 Mo == Yes")
                            had_prev_pref = (previous_preferential_rent_has_value == "Yes")
                            explanation_steps.append(f"6. Previous Preferential Rent had value? {'Yes' if had_prev_pref else 'No'}")
                            if had_prev_pref:
                                log.info("Path: Pref Rent Has Value == Yes")
                                result = 0.0
                                explanation_steps.append("--> Result: 0.00%")
                            else: # Pref Rent No
                                log.info("Path: Pref Rent Has Value == No")
                                result = 0.20 - (two_yr_rate - one_yr_rate)
                                explanation_steps.append(f"--> Calculation: 20.00% - (2-Year Rate ({two_yr_rate*100:.2f}%) - 1-Year Rate ({one_yr_rate*100:.2f}%)) = {result*100:.2f}%")
                        else: # Had Vacancy Allowance No
                            log.info("Path: Had Vacancy Allowance Prev 12 Mo == No")
                            tenure = tenant_tenure_years
                            log.info(f"Path: Tenant Tenure Check. Tenure = {tenure}")
                            explanation_steps.append(f"6. Checking Tenant Tenure: {tenure} years")
                            if 0 <= tenure <= 2:
                                result = 0.05
                                explanation_steps.append(f"--> Tenure ({tenure}) is 0-2 years. Result: 5.00%")
                            elif tenure == 3:
                                result = 0.10
                                explanation_steps.append(f"--> Tenure ({tenure}) is 3 years. Result: 10.00%")
                            elif tenure == 4:
                                result = 0.15
                                explanation_steps.append(f"--> Tenure ({tenure}) is 4 years. Result: 15.00%")
                            elif tenure > 4:
                                result = 0.20 - (two_yr_rate - one_yr_rate)
                                explanation_steps.append(f"--> Tenure ({tenure}) > 4 years.")
                                explanation_steps.append(f"--> Calculation: 20.00% - (2-Year Rate ({two_yr_rate*100:.2f}%) - 1-Year Rate ({one_yr_rate*100:.2f}%)) = {result*100:.2f}%")
                            else:
                                result = "Indeterminable: Invalid Tenure"
                                explanation_steps.append(f"--> Invalid Tenure value: {tenure}. Result is indeterminable.")
                    elif DATE_RANGES["range_19_present"][0] <= lsd: # Simplified check for present
                        log.info("Path: Lease Date 19-Present")
                        explanation_steps.append(f"--> Date falls within {DATE_RANGES['range_19_present'][0]} to present.")
                        result = 0.0
                        explanation_steps.append("--> Result: 0.00%")
                    else:
                        result = "Indeterminable: Lease Start Date out of defined ranges (Term 1)"
                        explanation_steps.append(f"--> Lease Start Date {lsd} does not fall into any defined range for Term 1.")

                elif term_length >= 2:
                    log.info("Path: Term Length == 2+")
                    explanation_steps.append(f"4. Checking Lease Start Date ({lsd}) for 2+ Year Term...")
                    # Check Lease Start Date (Term 2+)
                    if DATE_RANGES["range_83_97"][0] <= lsd <= DATE_RANGES["range_83_97"][1]:
                        log.info("Path: Lease Date 83-97")
                        explanation_steps.append(f"--> Date falls within {DATE_RANGES['range_83_97'][0]} to {DATE_RANGES['range_83_97'][1]}.")
                        result = 0.20 - two_yr_rate + vac_lease_rate
                        explanation_steps.append(f"--> Calculation: 20.00% - 2-Year Rate ({two_yr_rate*100:.2f}%) + Vacancy Lease Rate ({vac_lease_rate*100:.2f}%) = {result*100:.2f}%")
                    elif DATE_RANGES["range_97_11"][0] <= lsd <= DATE_RANGES["range_97_11"][1]:
                        log.info("Path: Lease Date 97-11")
                        explanation_steps.append(f"--> Date falls within {DATE_RANGES['range_97_11'][0]} to {DATE_RANGES['range_97_11'][1]}.")
                        result = 0.20
                        explanation_steps.append("--> Result: 20.00%")
                    elif DATE_RANGES["range_11_15"][0] <= lsd <= DATE_RANGES["range_11_15"][1]:
                        log.info("Path: Lease Date 11-15")
                        explanation_steps.append(f"--> Date falls within {DATE_RANGES['range_11_15'][0]} to {DATE_RANGES['range_11_15'][1]}.")
                        result = 0.0
                        explanation_steps.append("--> Result: 0.00%")
                    elif DATE_RANGES["range_15_19"][0] <= lsd <= DATE_RANGES["range_15_19"][1]:
                        log.info("Path: Lease Date 15-19")
                        explanation_steps.append(f"--> Date falls within {DATE_RANGES['range_15_19'][0]} to {DATE_RANGES['range_15_19'][1]}.")
                        had_prev_vac = (had_vacancy_allowance_in_prev_12_mo == "Yes")
                        explanation_steps.append(f"5. Vacancy Allowance taken in prior 12 mo? {'Yes' if had_prev_vac else 'No'}")
                        if had_prev_vac:
                            log.info("Path: Had Vacancy Allowance Prev 12 Mo == Yes")
                            result = 0.0
                            explanation_steps.append("--> Result: 0.00%")
                        else: # Had Vacancy Allowance No
                            log.info("Path: Had Vacancy Allowance Prev 12 Mo == No")
                            had_prev_pref = (previous_preferential_rent_has_value == "Yes")
                            explanation_steps.append(f"6. Previous Preferential Rent had value? {'Yes' if had_prev_pref else 'No'}")
                            if had_prev_pref:
                                log.info("Path: Pref Rent Has Value == Yes")
                                result = 0.0
                                explanation_steps.append("--> Result: 0.00%")
                            else: # Pref Rent No
                                log.info("Path: Pref Rent Has Value == No")
                                result = 0.20
                                explanation_steps.append("--> Result: 20.00%")
                    elif DATE_RANGES["range_19_present"][0] <= lsd: # Simplified check for present
                        log.info("Path: Lease Date 19-Present")
                        explanation_steps.append(f"--> Date falls within {DATE_RANGES['range_19_present'][0]} to present.")
                        had_prev_vac = (had_vacancy_allowance_in_prev_12_mo == "Yes")
                        explanation_steps.append(f"5. Vacancy Allowance taken in prior 12 mo? {'Yes' if had_prev_vac else 'No'}")
                        if had_prev_vac:
                            log.info("Path: Had Vacancy Allowance Prev 12 Mo == Yes")
                            result = 0.0
                            explanation_steps.append("--> Result: 0.00%")
                        else: # Had Vacancy Allowance No
                            log.info("Path: Had Vacancy Allowance Prev 12 Mo == No")
                            result = one_yr_rate
                            explanation_steps.append(f"--> Result: 1-Year Rate ({result*100:.2f}%)")
                    else:
                        result = "Indeterminable: Lease Start Date out of defined ranges (Term 2+)"
                        explanation_steps.append(f"--> Lease Start Date {lsd} does not fall into any defined range for Term 2+.")
                else:
                     result = "Indeterminable: Invalid Term Length" # Should not happen with Radio
                     explanation_steps.append(f"---> Invalid Term Length provided: {term_length_str}")

        formatted_result = format_result(result)
        # Use the pre-formatted result directly
        explanation_steps.append(f"\nFinal Result: {formatted_result}")

        log.info(f"--- Calculation Complete. Result: {formatted_result} ---")
        explanation_log = "\n".join(explanation_steps)
        notes_log = "\n".join(notes_details) # Compile notes log

        return formatted_result, explanation_log, notes_log # Return all three

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
def create_calculator_tab():
    """Creates the Gradio Blocks UI for the Vacancy Allowance Calculator tab."""
    with gr.Blocks() as calculator_tab:
        gr.Markdown("## Vacancy Allowance Calculator (Based on Flowchart)")
        gr.Markdown("Enter the details below as per the flowchart to calculate the allowance.")

        if rgb_data is None or rgb_data.is_empty():
             gr.Markdown("**Error: RGBO data failed to load. Calculator is non-functional.** Check application logs.", elem_id="error-message")

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


        # Event Handler
        calculate_button.click(
            fn=calculate_vacancy_allowance_interactive,
            inputs=[
                apt_status_input,
                new_tenant_input,
                term_length_input,
                lease_start_input,
                prev_vac_allow_input,
                prev_pref_rent_input,
                tenant_tenure_input
            ],
            outputs=[result_output, logic_output, notes_output] # Update outputs to include the new textbox and notes
        )

    # Return the Blocks object so it can be rendered in the main app
    return calculator_tab

# Example of running just this tab for testing (optional)
if __name__ == '__main__':
    print("Launching Vacancy Calculator Tab standalone for testing...")
    if rgb_data is None:
         print("\nWARNING: RGBO data failed to load. The calculator will show an error message and be disabled.\n")
    interface = create_calculator_tab()
    interface.launch() 