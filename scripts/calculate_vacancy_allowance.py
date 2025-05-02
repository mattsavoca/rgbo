import polars as pl
from datetime import datetime
import os
import logging # Import logging

# Configure logging specific to this module IF not configured globally
# This ensures logs appear if the module is run standalone or imported.
# If global config exists (e.g., in app.py or pdf_handler.py), this might be redundant but is safe.
log = logging.getLogger(__name__) # Use a module-specific logger
if not log.hasHandlers(): # Avoid adding handlers multiple times if already configured
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s] - %(message)s')


RGBO_CSV_PATH = "rgbo.csv" # Path to the Rent Guidelines Board Orders CSV

def load_rgb_orders(csv_path=RGBO_CSV_PATH):
    """Loads RGBO data, parses dates using Polars, and prepares it for lookup."""
    log.info(f"Attempting to load RGBO data from: {csv_path}") # Use logger
    if not os.path.exists(csv_path):
        log.error(f"RGBO data file not found at the specified path: {csv_path}") # Use logger
        raise FileNotFoundError(f"RGBO data file not found at: {csv_path}")

    try:
        # Use Polars read_csv
        df = pl.read_csv(csv_path)
        log.info(f"Successfully read CSV. Initial shape: {df.shape}") # Use logger

        # Convert date columns to datetime objects using Polars expressions
        log.info("Parsing date columns...") # Use logger
        df = df.with_columns([
            pl.col('beginning_date').str.strptime(pl.Date, "%m/%d/%Y", strict=False).alias('beginning_date'),
            pl.col('end_date').str.strptime(pl.Date, "%m/%d/%Y", strict=False).alias('end_date')
        ])
        log.info(f"Shape after date parsing attempt: {df.shape}") # Use logger

        # Convert rate columns to numeric (Float64), coercing errors to null
        rate_cols = ['one_year_rate', 'two_year_rate', 'vacancy_lease_rate']
        log.info("Parsing rate columns...") # Use logger
        for col in rate_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
            else:
                log.warning(f"Expected rate column '{col}' not found in {csv_path}. Filling with null.") # Use logger
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col)) # Add column with nulls

        # Drop rows where essential dates are missing (null) after parsing attempts
        log.info("Dropping rows with null beginning_date or end_date...") # Use logger
        original_rows = df.height
        df = df.drop_nulls(subset=['beginning_date', 'end_date'])
        rows_after_drop = df.height
        log.info(f"Shape after dropping null dates: {df.shape}. Dropped {original_rows - rows_after_drop} rows.") # Use logger

        if df.is_empty():
             log.warning(f"RGBO DataFrame is empty after processing (dropping null dates). Check date parsing for {csv_path}.") # Use logger

        # Sort by beginning_date descending
        df = df.sort(by='beginning_date', descending=True)

        log.info(f"Successfully loaded and processed {len(df)} RGBO records from {csv_path}") # Use logger
        return df
    except Exception as e:
        log.error(f"Error loading or processing RGBO data from {csv_path}: {e}", exc_info=True) # Use logger, add exc_info
        raise # Re-raise the exception after logging

def get_rates_for_date(rgb_data, relevant_date):
    """
    Finds the applicable RGBO rates for a given date using Polars.

    Args:
        rgb_data (pl.DataFrame): DataFrame loaded by load_rgb_orders.
        relevant_date (datetime.date): The date to find rates for.

    Returns:
        dict or None: A dictionary containing rates for that date, or None if not found.
    """
    if relevant_date is None: # Check if date is None (previously pd.isna)
        return None

    # Convert relevant_date to the same type as DataFrame columns if needed, assuming date object
    relevant_date_pl = relevant_date # Polars can often compare directly with Python date/datetime

    # Find the first row where the relevant_date falls within the order's range
    applicable_order_df = rgb_data.filter(
        (pl.col('beginning_date') <= relevant_date_pl) &
        (pl.col('end_date') >= relevant_date_pl)
    )

    if applicable_order_df.height > 0:
        # Handle multiple matching orders if necessary (e.g., tenant vs owner paid heat)
        # For now, just return the first match as a dictionary.
        return applicable_order_df.row(0, named=True)
    else:
        return None

# --- Define Date Ranges from Flowchart ---
# Using date objects for comparison (Polars works well with date/datetime)
# Assuming the CSV dates are parsed as date objects by Polars' strptime
DATE_RANGES = {
    "range_83_97": (datetime(1983, 6, 19).date(), datetime(1997, 6, 18).date()),
    "range_97_11": (datetime(1997, 6, 19).date(), datetime(2011, 6, 23).date()),
    "range_11_15": (datetime(2011, 6, 24).date(), datetime(2015, 6, 14).date()),
    "range_15_19": (datetime(2015, 6, 15).date(), datetime(2019, 6, 13).date()),
    "range_19_present": (datetime(2019, 6, 14).date(), datetime.now().date()) # Use current date as end
}

def calculate_vacancy_allowance(unit_data, rgb_data):
    """
    Calculates the vacancy allowance based on the flowchart logic using Polars data.

    Args:
        unit_data (dict): A dictionary containing data for a single unit.
                          Expected keys: 'apartment_status', 'is_new_tenant',
                          'term_length', 'lease_start_date' (datetime.date obj),
                          'had_vacancy_allowance_in_prev_12_mo',
                          'previous_preferential_rent_has_value',
                          'tenant_tenure_years',
        rgb_data (pl.DataFrame): Loaded RGBO data (Polars DataFrame).

    Returns:
        float or str: The calculated vacancy allowance percentage (e.g., 0.05 for 5%)
                      or a descriptive string if a rate is indeterminable.
                      Returns None if input data is insufficient.
    """
    # --- Input Validation ---
    required_keys = ['apartment_status', 'is_new_tenant', 'term_length', 'lease_start_date',
                     'had_vacancy_allowance_in_prev_12_mo', 'previous_preferential_rent_has_value',
                     'tenant_tenure_years']
    if not all(key in unit_data for key in required_keys):
        print("Warning: Missing required keys in unit_data")
        return None
    # Use standard None check
    if unit_data['lease_start_date'] is None:
         print("Warning: Missing lease_start_date in unit_data")
         return None # Cannot proceed without lease start date

    # --- Get Relevant Rates ---
    # Convert lease_start_date to date object if it's datetime for comparison
    lease_start_dt = unit_data['lease_start_date']
    lease_start_date_obj = lease_start_dt.date() if isinstance(lease_start_dt, datetime) else lease_start_dt

    rates = get_rates_for_date(rgb_data, lease_start_date_obj) # Pass date object
    if rates is None:
        # Use f-string formatting or str() for the date
        return f"Indeterminable: No RGBO rates found for lease start date {str(lease_start_date_obj)}"

    # Rates is now a dictionary
    one_yr_rate = rates.get('one_year_rate', 0.0)
    two_yr_rate = rates.get('two_year_rate', 0.0)
    vac_lease_rate = rates.get('vacancy_lease_rate', 0.0) # May not exist in all orders

    # Fill None rates (Polars uses None for nulls) with 0.0
    one_yr_rate = 0.0 if one_yr_rate is None else one_yr_rate
    two_yr_rate = 0.0 if two_yr_rate is None else two_yr_rate
    vac_lease_rate = 0.0 if vac_lease_rate is None else vac_lease_rate


    # --- Flowchart Logic --- Helper function for duplicated logic ---
    def _calculate_term1_or_existing_allowance(lsd, unit_data, one_yr_rate, two_yr_rate, vac_lease_rate):
        """Helper function for logic shared by New Tenant (Term 1) and Existing Tenant."""
        if DATE_RANGES["range_83_97"][0] <= lsd <= DATE_RANGES["range_83_97"][1]:
             # Flowchart: [one_year_renewal_rate] - [vacancy_lease_rate]
             return (one_yr_rate - vac_lease_rate)
        elif DATE_RANGES["range_97_11"][0] <= lsd <= DATE_RANGES["range_97_11"][1]:
             # Flowchart: 20% - [two_year_renewal_rate] - [one_year_renewal]
             return 0.20 - (two_yr_rate - one_yr_rate)
        elif DATE_RANGES["range_11_15"][0] <= lsd <= DATE_RANGES["range_11_15"][1]:
            if unit_data['had_vacancy_allowance_in_prev_12_mo']:
                 if unit_data['previous_preferential_rent_has_value']:
                     # Flowchart: 20% - [two_year_renewal_rate] - [one_year_renewal]
                     return 0.20 - (two_yr_rate - one_yr_rate)
                 else:
                      # Flowchart: 20% - [two_year_renewal_rate]
                      return 0.20 - two_yr_rate
            else: # No vacancy allowance in prev 12 mo -> Tenant Tenure
                tenure = unit_data['tenant_tenure_years']
                if 0 <= tenure <= 2: return 0.05
                elif tenure == 3: return 0.10
                elif tenure == 4: return 0.15
                elif tenure > 4:
                    # Flowchart: 20% - [two_year_renewal_rate] - [one_year_renewal]
                    return 0.20 - (two_yr_rate - one_yr_rate)
                else: return "Indeterminable: Invalid Tenure"
        elif DATE_RANGES["range_15_19"][0] <= lsd <= DATE_RANGES["range_15_19"][1]:
             if unit_data['had_vacancy_allowance_in_prev_12_mo']:
                 if unit_data['previous_preferential_rent_has_value']:
                      # Flowchart: 20%
                       return 0.20
                 else:
                      # Flowchart: 20%
                      return 0.20
             else: # No vacancy allowance in prev 12 mo -> Tenant Tenure
                 tenure = unit_data['tenant_tenure_years']
                 if 0 <= tenure <= 2: return 0.05
                 elif tenure == 3: return 0.10
                 elif tenure == 4: return 0.15
                 elif tenure > 4: return 0.20
                 else: return "Indeterminable: Invalid Tenure"
        elif DATE_RANGES["range_19_present"][0] <= lsd <= DATE_RANGES["range_19_present"][1]:
            # Flowchart shows same outcome regardless of prev 12 mo allowance
            return one_yr_rate
        else:
             return "Indeterminable: Lease Start Date out of defined ranges"

    # --- Main Flowchart Logic ---
    lsd = lease_start_date_obj # Use the date object for comparisons

    if unit_data['apartment_status'] == "PE":
        return one_yr_rate

    # Assuming 'Not PE' leads to New Tenant check
    if unit_data['is_new_tenant']:
        # --- New Tenant Path ---
        if unit_data['term_length'] == 1:
            # --- Use Helper for Term Length 1 ---
            result = _calculate_term1_or_existing_allowance(lsd, unit_data, one_yr_rate, two_yr_rate, vac_lease_rate)
            # Append specific context to indeterminable result if needed
            return result if not isinstance(result, str) else f"{result} (New Tenant, Term 1)"

        elif unit_data['term_length'] >= 2:
             # --- Term Length 2+ ---
            if DATE_RANGES["range_83_97"][0] <= lsd <= DATE_RANGES["range_83_97"][1]:
                 # Flowchart: 20% - [two_year_renewal_rate] + [vacancy_lease_rate]
                 # Interpretation: Subtract two_yr, add vac_lease. CHECK THIS ASSUMPTION.
                 return 0.20 - (two_yr_rate + vac_lease_rate)
            elif DATE_RANGES["range_97_11"][0] <= lsd <= DATE_RANGES["range_97_11"][1]:
                 return 0.20
            elif DATE_RANGES["range_11_15"][0] <= lsd <= DATE_RANGES["range_11_15"][1]:
                 return 0.20
            elif DATE_RANGES["range_15_19"][0] <= lsd <= DATE_RANGES["range_15_19"][1]:
                if unit_data['had_vacancy_allowance_in_prev_12_mo']:
                     return one_yr_rate
                else: # No vacancy allowance in prev 12 mo -> Tenant Tenure
                     tenure = unit_data['tenant_tenure_years']
                     if 0 <= tenure <= 2: return 0.05
                     elif tenure == 3: return 0.10
                     elif tenure == 4: return 0.15
                     elif tenure > 4: return 0.20
                     else: return "Indeterminable: Invalid Tenure"
            elif DATE_RANGES["range_19_present"][0] <= lsd <= DATE_RANGES["range_19_present"][1]:
                 # Flowchart shows same outcome regardless of prev 12 mo allowance
                 return one_yr_rate
            else:
                 return "Indeterminable: Lease Start Date out of defined ranges (Term 2+)"
        else:
             return "Indeterminable: Invalid Term Length"
    else:
        # --- Existing Tenant Path ---
        # Use Helper for Existing Tenant (follows Term 1 logic per flowchart)
        result = _calculate_term1_or_existing_allowance(lsd, unit_data, one_yr_rate, two_yr_rate, vac_lease_rate)
        # Append specific context to indeterminable result if needed
        return result if not isinstance(result, str) else f"{result} (Existing Tenant)"


# --- Example Usage ---
if __name__ == "__main__":
    # Define the path to the sample CSV in the root directory
    SAMPLE_CSV_PATH = "apt_sample.csv" # Assuming it's in the same dir as rgbo.csv

    print("Loading RGBO data...")
    try:
        # Load RGBO data first, as it's needed for calculations
        rgb_order_data = load_rgb_orders() # Uses the default RGBO_CSV_PATH

        print(f"Loading sample unit data from: {SAMPLE_CSV_PATH}")
        if not os.path.exists(SAMPLE_CSV_PATH):
             print(f"Error: Sample CSV file not found at {SAMPLE_CSV_PATH}")
             exit() # Exit if sample file is missing

        # Load the sample CSV
        df = pl.read_csv(SAMPLE_CSV_PATH, try_parse_dates=True)

        if df.is_empty():
            print(f"Sample CSV file {SAMPLE_CSV_PATH} is empty. Exiting.")
            exit()

        print(f"Preprocessing sample data (replicating pdf_handler logic)...")

        # --- Replicate Feature Engineering from pdf_handler.py ---
        required_source_cols = ["Lease Began", "Lease Ends", "Legal Reg Rent", "Actual Rent Paid", "Apt Status", "Tenant Name"]
        missing_cols = [col for col in required_source_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Sample CSV {SAMPLE_CSV_PATH} is missing required columns: {missing_cols}")
            exit()

        # Ensure correct types after load
        df = df.with_columns([
            pl.col("Lease Began").cast(pl.Date, strict=False),
            pl.col("Lease Ends").cast(pl.Date, strict=False),
            pl.col("Legal Reg Rent").cast(pl.Float64, strict=False),
            pl.col("Actual Rent Paid").cast(pl.Float64, strict=False),
            pl.col("Apt Status").cast(pl.String, strict=False).fill_null("Unknown"),
            pl.col("Tenant Name").cast(pl.String, strict=False).fill_null("Unknown Tenant")
        ])

        # Sort by lease start date
        df = df.sort("Lease Began", nulls_last=True)

        # Feature Engineering expressions
        df = df.with_columns([
            pl.when(pl.col("Tenant Name").shift(1).is_null() | (pl.col("Tenant Name") != pl.col("Tenant Name").shift(1)))
              .then(pl.lit(True))
              .otherwise(pl.lit(False))
              .alias("is_new_tenant"),
            pl.when(pl.col("Lease Ends").is_not_null() & pl.col("Lease Began").is_not_null())
              .then(
                  pl.when((pl.col("Lease Ends") - pl.col("Lease Began")).dt.total_days() > 548)
                  .then(pl.lit(2))
                  .otherwise(pl.lit(1))
              )
              .otherwise(pl.lit(1))
              .alias("term_length"),
            pl.col("Lease Began").alias("lease_start_date"),
            pl.lit(False).alias("had_vacancy_allowance_in_prev_12_mo"), # Placeholder
            pl.when(
                pl.col("Actual Rent Paid").shift(1).is_not_null() &
                pl.col("Legal Reg Rent").shift(1).is_not_null() &
                (pl.col("Actual Rent Paid").shift(1) < pl.col("Legal Reg Rent").shift(1))
             )
              .then(pl.lit(True))
              .otherwise(pl.lit(False))
              .fill_null(False)
              .alias("previous_preferential_rent_has_value"),
        ])

        # Tenant Tenure Calculation
        df = df.with_columns(
            pl.col("Tenant Name").rle_id().alias("tenant_block_id")
        )
        df = df.with_columns(
            pl.when(pl.col("Lease Began").is_not_null())
            .then(
                ((pl.col("Lease Began") - pl.col("Lease Began").first().over("tenant_block_id")).dt.total_days() / 365.25)
                .floor()
                .cast(pl.Int64)
            )
            .otherwise(pl.lit(0))
            .alias("tenant_tenure_years")
        ).drop("tenant_block_id")
        # --- End of Replicated Logic ---

        print("\nCalculating vacancy allowances for preprocessed sample data:")
        # Convert DataFrame rows to list of dictionaries for processing
        unit_data_list = df.to_dicts()

        for i, unit_data_row in enumerate(unit_data_list):
            # Prepare the unit_data dict expected by the function
            # Note: The keys in unit_data_row match the DataFrame columns now
            unit_data_for_calc = {
                'apartment_status': unit_data_row.get("Apt Status"),
                'is_new_tenant': unit_data_row.get("is_new_tenant"),
                'term_length': unit_data_row.get("term_length"),
                'lease_start_date': unit_data_row.get("lease_start_date"), # Already a date object
                'had_vacancy_allowance_in_prev_12_mo': unit_data_row.get("had_vacancy_allowance_in_prev_12_mo"),
                'previous_preferential_rent_has_value': unit_data_row.get("previous_preferential_rent_has_value"),
                'tenant_tenure_years': unit_data_row.get("tenant_tenure_years")
            }

            # Debug print the data being passed FOR THE FIRST ROW ONLY
            if i == 0:
                print("\n--- Data for first row passed to calculate_vacancy_allowance: ---")
                for key, value in unit_data_for_calc.items():
                    print(f"  {key}: {value} (Type: {type(value)})")
                print("----------------------------------------------------------------")


            allowance = calculate_vacancy_allowance(unit_data_for_calc, rgb_order_data)
            # Include original Lease Began for context in output
            original_lease_began = unit_data_row.get("Lease Began")
            print(f"  Row {i+1} (Lease Began: {original_lease_began}): Calculated Allowance = {allowance}")

    except FileNotFoundError as e:
        print(f"Error: {e}") # Handles RGBO file not found too
    except Exception as e:
        # Use logger if available, otherwise print
        log.error(f"An error occurred during example execution: {e}", exc_info=True) if 'log' in locals() else print(f"An error occurred during example execution: {e}")
        # Consider printing traceback explicitly if logger isn't set up in __main__
        import traceback
        traceback.print_exc() 