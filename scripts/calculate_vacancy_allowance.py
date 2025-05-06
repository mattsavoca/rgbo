import polars as pl
from datetime import datetime, date # Ensure date is imported
import os
import logging

# Configure logging
log = logging.getLogger(__name__)
if not log.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s] - %(message)s')

RGBO_CSV_PATH = "rgbo.csv"

# --- DATE_RANGES definition should be at the module level ---
DATE_RANGES = {
    "range_83_97": (datetime(1983, 6, 19).date(), datetime(1997, 6, 18).date()),
    "range_97_11": (datetime(1997, 6, 19).date(), datetime(2011, 6, 23).date()),
    "range_11_15": (datetime(2011, 6, 24).date(), datetime(2015, 6, 14).date()),
    "range_15_19": (datetime(2015, 6, 15).date(), datetime(2019, 6, 13).date()),
    "range_19_present": (datetime(2019, 6, 14).date(), datetime.now().date())
}

def load_rgb_orders(csv_path=RGBO_CSV_PATH):
    """Loads RGBO data, parses dates using Polars, and prepares it for lookup."""
    log.info(f"Attempting to load RGBO data from: {csv_path}")
    if not os.path.exists(csv_path):
        log.error(f"RGBO data file not found at the specified path: {csv_path}")
        raise FileNotFoundError(f"RGBO data file not found at: {csv_path}")

    try:
        df = pl.read_csv(csv_path)
        log.info(f"Successfully read CSV. Initial shape: {df.shape}")

        date_format_primary = "%m/%d/%Y"
        date_format_fallback = "%Y-%m-%d" # Added fallback

        df = df.with_columns(
            pl.coalesce(
                pl.col('beginning_date').str.strptime(pl.Date, date_format_primary, strict=False),
                pl.col('beginning_date').str.strptime(pl.Date, date_format_fallback, strict=False)
            ).alias('beginning_date'),
            pl.coalesce(
                pl.col('end_date').str.strptime(pl.Date, date_format_primary, strict=False),
                pl.col('end_date').str.strptime(pl.Date, date_format_fallback, strict=False)
            ).alias('end_date')
        )
        log.info(f"Shape after date parsing attempt: {df.shape}")

        rate_cols = [
            'one_year_rate', 'two_year_rate', 'vacancy_lease_rate',
            'two_year_first_year_rate', 'two_year_second_year_rate',
            'one_year_first_half_rate', 'one_year_second_half_rate'
        ]
        log.info("Parsing rate columns...")
        for col_name in rate_cols:
            if col_name in df.columns:
                df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False))
            else:
                log.warning(f"Expected rate column '{col_name}' not found in {csv_path}. Will be treated as null/0.0 where applicable.")

        original_rows = df.height
        df = df.drop_nulls(subset=['beginning_date', 'end_date'])
        rows_after_drop = df.height
        log.info(f"Shape after dropping null dates: {df.shape}. Dropped {original_rows - rows_after_drop} rows.")

        if df.is_empty():
             log.warning(f"RGBO DataFrame is empty after processing. Check date parsing for {csv_path}.")

        df = df.sort(by='beginning_date', descending=True)
        log.info(f"Successfully loaded and processed {len(df)} RGBO records from {csv_path}")
        return df
    except Exception as e:
        log.error(f"Error loading or processing RGBO data from {csv_path}: {e}", exc_info=True)
        raise

def get_order_details_for_date(rgb_data: pl.DataFrame, relevant_date: date) -> dict | None:
    """
    Finds the applicable RGBO order details for a given date.
    """
    if relevant_date is None:
        return None

    applicable_order_df = rgb_data.filter(
        (pl.col('beginning_date') <= relevant_date) &
        (pl.col('end_date') >= relevant_date)
    )

    if applicable_order_df.height > 0:
        if applicable_order_df.height > 1:
            log.warning(f"Multiple RGBO orders found for {relevant_date}. Using the first one: {applicable_order_df[0, 'order_number']}")
        return applicable_order_df.row(0, named=True)
    else:
        return None

def calculate_vacancy_allowance_for_row(
    unit_data_current_row: dict,
    rgb_data_full: pl.DataFrame,
    previous_row_info: dict | None
) -> float | str:
    """
    Calculates the vacancy allowance for a single unit's lease record,
    incorporating logic from the interactive calculator.
    """
    lsd = unit_data_current_row.get('lease_start_date')

    # --- Updated Check Order: Check for None FIRST --- 
    if lsd is None:
        # Imputation failed or wasn't possible
        return "Indeterminable: Missing lease_start_date" 

    # Now check if it's a valid date type if not None
    if not isinstance(lsd, date):
        # Attempt conversion if it's datetime (should have been handled by caller, but belt-and-suspenders)
        if isinstance(lsd, datetime):
            lsd = lsd.date()
        else:
            # If it's still not a date object after checking None and datetime, it's an invalid type.
            log.error(f"lease_start_date is not a valid Python date object: {lsd} (type: {type(lsd)}) after None check.")
            return "Indeterminable: Invalid lease_start_date type"
    
    order_details = get_order_details_for_date(rgb_data_full, lsd)
    if order_details is None:
        return f"Indeterminable: No RGBO order found for lease start date {lsd}"

    one_yr_rate = float(order_details.get('one_year_rate', 0.0) or 0.0)
    two_yr_rate = float(order_details.get('two_year_rate', 0.0) or 0.0)
    vac_lease_rate_val = order_details.get('vacancy_lease_rate')
    vac_lease_rate = float(vac_lease_rate_val or 0.0)
    order_number = str(order_details.get('order_number', 'Unknown'))
    two_year_first_year_rate = float(order_details.get('two_year_first_year_rate', 0.0) or 0.0)
    two_year_second_year_rate = float(order_details.get('two_year_second_year_rate', 0.0) or 0.0)
    one_year_first_half_rate = float(order_details.get('one_year_first_half_rate', 0.0) or 0.0)
    one_year_second_half_rate = float(order_details.get('one_year_second_half_rate', 0.0) or 0.0)

    apartment_status_str = unit_data_current_row.get('Apt Status', "Other")
    is_new_tenant_bool = unit_data_current_row.get('is_new_tenant', False)
    is_new_tenant_str = "Yes" if is_new_tenant_bool else "No"
    term_length_int = unit_data_current_row.get('term_length', 1)
    term_length_str = "2+" if term_length_int >= 2 else "1"

    had_vacancy_allowance_bool = False
    if previous_row_info and previous_row_info.get('tenant_name') == unit_data_current_row.get('Tenant Name'):
        prev_allowance = previous_row_info.get('x_vacancy_allowance', 0.0)
        prev_lease_end = previous_row_info.get('lease_end_date')
        if isinstance(prev_allowance, (float, int)) and prev_allowance > 0 and isinstance(prev_lease_end, date):
            if lsd and (lsd - prev_lease_end).days <= 366:
                 had_vacancy_allowance_bool = True
    had_vacancy_allowance_in_prev_12_mo_str = "Yes" if had_vacancy_allowance_bool else "No"

    prev_pref_rent_bool = unit_data_current_row.get('previous_preferential_rent_has_value', False)
    previous_preferential_rent_has_value_str = "Yes" if prev_pref_rent_bool else "No"
    tenant_tenure_years = float(unit_data_current_row.get('tenant_tenure_years', 0.0))

    is_order52_first_year_str = "No"
    if order_number == "52" and term_length_str == "2+":
        is_order52_first_year_str = "Yes"

    is_order53_first_half_str = "Second 6 Months"
    if order_number == "53" and term_length_str == "1":
        is_order53_first_half_str = "First 6 Months"

    result_val: float | str
    is_pe_status = (apartment_status_str == "PE")

    if is_pe_status:
        result_val = one_yr_rate
    else:
        is_new = (is_new_tenant_str == "Yes")
        if not is_new:
            result_val = 0.0
        else:
            if term_length_str == "1":
                applicable_one_year_rate = one_yr_rate
                if order_number == "53":
                    if is_order53_first_half_str == "First 6 Months":
                        applicable_one_year_rate = one_year_first_half_rate
                    else:
                        applicable_one_year_rate = one_year_second_half_rate

                if DATE_RANGES["range_83_97"][0] <= lsd <= DATE_RANGES["range_83_97"][1]:
                    result_val = applicable_one_year_rate + vac_lease_rate
                elif DATE_RANGES["range_97_11"][0] <= lsd <= DATE_RANGES["range_97_11"][1]:
                    result_val = 0.20 - (two_yr_rate - applicable_one_year_rate)
                elif DATE_RANGES["range_11_15"][0] <= lsd <= DATE_RANGES["range_11_15"][1]:
                    result_val = 0.0
                elif DATE_RANGES["range_15_19"][0] <= lsd <= DATE_RANGES["range_15_19"][1]:
                    had_prev_vac = (had_vacancy_allowance_in_prev_12_mo_str == "Yes")
                    if had_prev_vac:
                        had_prev_pref = (previous_preferential_rent_has_value_str == "Yes")
                        if had_prev_pref:
                            result_val = 0.0
                        else:
                            result_val = 0.20 - (two_yr_rate - applicable_one_year_rate)
                    else:
                        tenure = tenant_tenure_years
                        if 0 <= tenure <= 2: result_val = 0.05
                        elif tenure == 3: result_val = 0.10
                        elif tenure == 4: result_val = 0.15
                        elif tenure > 4: result_val = 0.20 - (two_yr_rate - applicable_one_year_rate)
                        else: result_val = "Indeterminable: Invalid Tenure (1-yr)"
                elif DATE_RANGES["range_19_present"][0] <= lsd:
                    result_val = applicable_one_year_rate
                else:
                    result_val = "Indeterminable: Lease Start Date out of defined ranges (Term 1)"

            elif term_length_str == "2+":
                applicable_two_year_rate = two_yr_rate
                if order_number == "52":
                    if is_order52_first_year_str == "Yes":
                        applicable_two_year_rate = two_year_first_year_rate
                    else:
                        applicable_two_year_rate = two_year_second_year_rate
                
                if DATE_RANGES["range_83_97"][0] <= lsd <= DATE_RANGES["range_83_97"][1]:
                    result_val = 0.20 - applicable_two_year_rate + vac_lease_rate
                elif DATE_RANGES["range_97_11"][0] <= lsd <= DATE_RANGES["range_97_11"][1]:
                    result_val = 0.20
                elif DATE_RANGES["range_11_15"][0] <= lsd <= DATE_RANGES["range_11_15"][1]:
                    result_val = 0.0
                elif DATE_RANGES["range_15_19"][0] <= lsd <= DATE_RANGES["range_15_19"][1]:
                    had_prev_vac = (had_vacancy_allowance_in_prev_12_mo_str == "Yes")
                    if had_prev_vac:
                        result_val = 0.0
                    else: 
                        had_prev_pref = (previous_preferential_rent_has_value_str == "Yes")
                        if had_prev_pref:
                            result_val = 0.0
                        else: 
                            result_val = 0.20
                elif DATE_RANGES["range_19_present"][0] <= lsd: 
                    had_prev_vac = (had_vacancy_allowance_in_prev_12_mo_str == "Yes")
                    if had_prev_vac:
                        result_val = 0.0
                    else: 
                        result_val = applicable_two_year_rate
                else:
                    result_val = "Indeterminable: Lease Start Date out of defined ranges (Term 2+)"
            else:
                 result_val = "Indeterminable: Invalid Term Length"
    
    if isinstance(result_val, (int, float)):
        return float(result_val)
    return str(result_val)

def calculate_single_allowance_from_dict(unit_data_dict: dict, rgb_data_full: pl.DataFrame) -> float | str:
    """
    Wrapper function for the interactive calculator in the UI.
    Calls calculate_vacancy_allowance_for_row with no previous_row_info.
    Ensures unit_data_dict has 'lease_start_date' as a Python date object.
    """
    # Ensure lease_start_date is a Python date object before calling the main logic
    # This mirrors the conversion that happens in add_vacancy_allowances_to_df's loop
    lsd = unit_data_dict.get('lease_start_date')
    if lsd is not None and not isinstance(lsd, date):
        if isinstance(lsd, datetime):
            unit_data_dict['lease_start_date'] = lsd.date()
        # Add handling for other potential types if necessary, or ensure Gradio provides date object
        elif isinstance(lsd, str): # Example if Gradio sometimes passes string
            try:
                unit_data_dict['lease_start_date'] = datetime.strptime(lsd, "%Y-%m-%d").date()
            except ValueError:
                log.error(f"calculate_single_allowance_from_dict: Could not parse lease_start_date string: {lsd}")
                return "Indeterminable: Invalid lease_start_date string format"
        # If it's already a Polars date from a direct DF read, it might be handled by the main func or need to_pydate()
        elif hasattr(lsd, 'to_pydate'): # Check for Polars date
             unit_data_dict['lease_start_date'] = lsd.to_pydate()
        # else: it might be an unexpected type, calculate_vacancy_allowance_for_row will handle it

    # Default other potentially missing fields that calculate_vacancy_allowance_for_row expects
    # from the feature engineering steps if they are not already in unit_data_dict
    # For the interactive tab, these are direct inputs, so they should exist.
    # We rely on calculate_vacancy_allowance_for_row's .get() for robustness for most fields.

    return calculate_vacancy_allowance_for_row(
        unit_data_current_row=unit_data_dict, 
        rgb_data_full=rgb_data_full, 
        previous_row_info=None
    )

def add_vacancy_allowances_to_df(unit_df: pl.DataFrame, rgb_data_full: pl.DataFrame) -> pl.DataFrame:
    """
    Processes a unit's DataFrame to add engineered features and an 'x_vacancy_allowance' column.

    Args:
        unit_df: The input Polars DataFrame for a single unit, expected to have columns like
                 'Lease Began', 'Lease Ends', 'Tenant Name', 'Actual Rent Paid', 'Legal Reg Rent', etc.
        rgb_data_full: The fully loaded and processed RGBO Polars DataFrame.

    Returns:
        A new Polars DataFrame with added columns including 'x_vacancy_allowance'.
    """
    log.info(f"Starting to process DataFrame for vacancy allowances. Input shape: {unit_df.shape}")
    df = unit_df.clone() # Work on a copy

    # --- Ensure date columns are parsed (if not already) ---
    if "Lease Began" in df.columns and df["Lease Began"].dtype == pl.String:
        df = df.with_columns(pl.col("Lease Began").str.strptime(pl.Date, "%m/%d/%Y", strict=False).alias("Lease Began"))
    elif "Lease Began" not in df.columns:
        log.warning("'Lease Began' column missing. Imputation and calculations might be affected.")
        df = df.with_columns(pl.lit(None, dtype=pl.Date).alias("Lease Began"))
        
    if "Lease Ends" in df.columns and df["Lease Ends"].dtype == pl.String:
        df = df.with_columns(pl.col("Lease Ends").str.strptime(pl.Date, "%m/%d/%Y", strict=False).alias("Lease Ends"))
    elif "Lease Ends" not in df.columns:
        log.warning("'Lease Ends' column missing. Imputation might be affected.")
        df = df.with_columns(pl.lit(None, dtype=pl.Date).alias("Lease Ends"))

    # --- Feature Engineering (Initial pass for is_new_tenant and lease_start_date alias) ---
    log.info("Performing initial feature engineering on the DataFrame...")
    df = df.sort("Lease Began", nulls_last=True) 
    
    if "Tenant Name" not in df.columns:
        log.warning("'Tenant Name' column missing. Creating a dummy tenant name for processing.")
        df = df.with_columns(pl.lit("UnknownTenant").alias("Tenant Name"))

    df = df.with_columns([
        pl.col("Lease Began").alias("lease_start_date"), # Alias first
        pl.when(pl.col("Tenant Name").shift(1).is_null() | (pl.col("Tenant Name") != pl.col("Tenant Name").shift(1)))
          .then(pl.lit(True))
          .otherwise(pl.lit(False))
          .alias("is_new_tenant")
    ])

    # --- Impute lease_start_date (Best Effort) ---
    log.info("Attempting to impute missing 'lease_start_date' values...")
    df = df.with_columns(
        pl.lit(False).alias("lease_start_date_imputed") # Initialize imputation tracking column
    )
    # Condition for imputation: lease_start_date is null, not a new tenant, and previous lease_end_date exists
    # We need previous row's Lease Ends for the same tenant.
    # This requires careful window function usage or an iterative approach if complex.
    # For a Polars-idiomatic approach for a simple forward fill based on previous lease end:
    df = df.with_columns(
        pl.col("Lease Ends").shift(1).over("Tenant Name").alias("prev_lease_end_for_impute")
    )

    imputation_condition = (
        pl.col("lease_start_date").is_null() & 
        pl.col("is_new_tenant").eq(False) & 
        pl.col("prev_lease_end_for_impute").is_not_null()
    )

    df = df.with_columns([
        pl.when(imputation_condition)
          .then(pl.col("prev_lease_end_for_impute") + pl.duration(days=1))
          .otherwise(pl.col("lease_start_date"))
          .alias("lease_start_date"),
        pl.when(imputation_condition)
          .then(pl.lit(True))
          .otherwise(pl.col("lease_start_date_imputed"))
          .alias("lease_start_date_imputed")
    ])
    imputed_count = df.filter(pl.col("lease_start_date_imputed")).height
    if imputed_count > 0:
        log.info(f"Imputed {imputed_count} missing 'lease_start_date' values.")
    df = df.drop("prev_lease_end_for_impute")
    # Re-sort if imputation changed dates that affect order (though imputation fills nulls, so order should be okay)
    # df = df.sort("lease_start_date", nulls_last=True)

    # --- Robust Casting for Rent Columns BEFORE comparison ---
    rent_cols_to_cast = ["Actual Rent Paid", "Legal Reg Rent"]
    for col_name in rent_cols_to_cast:
        if col_name in df.columns:
            # First, ensure it's string type to use string operations, then clean and cast
            if df[col_name].dtype == pl.String:
                df = df.with_columns(
                    pl.col(col_name)
                    .str.replace_all(r"[^\d\.]", "") # Remove anything not a digit or decimal point
                    .cast(pl.Float64, strict=False) # Cast to float, non-strict will turn errors to null
                    .alias(col_name)
                )
            elif df[col_name].dtype != pl.Float64: # If it's some other non-float numeric, just cast
                df = df.with_columns(
                    pl.col(col_name).cast(pl.Float64, strict=False).alias(col_name)
                )
            # If already Float64, do nothing
        else:
            log.warning(f"Rent column '{col_name}' not found in DataFrame. 'previous_preferential_rent_has_value' might be inaccurate.")
            # Add a null column of Float64 type if it's missing, so subsequent operations don't fail
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(col_name))

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

    df = df.with_columns(
        pl.col("Tenant Name").rle_id().alias("tenant_block_id")
    )
    df = df.with_columns(
    pl.when(pl.col("lease_start_date").is_not_null())
        .then(
        ((pl.col("lease_start_date") - pl.col("lease_start_date").first().over("tenant_block_id")).dt.total_days() / 365.25)
            .floor()
            .cast(pl.Int64)
        )
        .otherwise(pl.lit(0))
        .alias("tenant_tenure_years")
    ).drop("tenant_block_id")
    log.info("Feature engineering complete.")

    # --- Iterative Calculation ---
    log.info("Starting iterative vacancy allowance calculation for each row...")
    calculated_allowances = []
    previous_row_processed_info: dict | None = None

    for row_tuple in df.iter_rows(named=True):
        current_row_data_dict = dict(row_tuple)

        start_date_val = current_row_data_dict.get('lease_start_date')
        if hasattr(start_date_val, 'to_pydate'):
             current_row_data_dict['lease_start_date'] = start_date_val.to_pydate()
        
        end_date_val = current_row_data_dict.get('Lease Ends') # Original column name
        if hasattr(end_date_val, 'to_pydate'):
             current_row_data_dict['Lease Ends'] = end_date_val.to_pydate()

        allowance = calculate_vacancy_allowance_for_row(
            current_row_data_dict,
            rgb_data_full,
            previous_row_processed_info
        )
        calculated_allowances.append(allowance)

        current_allowance_for_prev_info = 0.0
        if isinstance(allowance, (float,int)):
            current_allowance_for_prev_info = float(allowance)

        previous_row_processed_info = {
            'x_vacancy_allowance': current_allowance_for_prev_info,
            'lease_end_date': current_row_data_dict.get('Lease Ends'), # Uses the potentially converted Python date
            'tenant_name': current_row_data_dict.get('Tenant Name')
        }
    log.info("Iterative calculation complete.")

    # Ensure x_vacancy_allowance is object type if it contains strings like "Indeterminable..."
    # Convert actual numeric results to float, keep strings as is, map Python None to Polars null
    processed_allowances = [
        float(x) if isinstance(x, (int, float)) 
        else (None if x is None or (isinstance(x, str) and x.lower() == 'none') 
              else str(x)) 
        for x in calculated_allowances
    ]
    allowance_series = pl.Series("x_vacancy_allowance", processed_allowances, dtype=pl.Object)
    df = df.with_columns(allowance_series)
    log.info(f"Added 'x_vacancy_allowance' column. Output shape: {df.shape}")

    # --- Calculate Rent Increase Percentage ---
    log.info("Calculating rent increase percentage...")
    if "Legal Reg Rent" in df.columns and df["Legal Reg Rent"].dtype == pl.Float64:
        df = df.with_columns(
            pl.col("Legal Reg Rent").shift(1).over("Tenant Name").alias("prev_legal_reg_rent")
        )
        
        # Define conditions separately for clarity and potential debugging
        cond1 = pl.col("Legal Reg Rent").is_not_null()
        cond2 = pl.col("prev_legal_reg_rent").is_not_null()
        cond3 = (pl.col("prev_legal_reg_rent") != 0) # Wrap comparison
        
        # Combine boolean conditions
        # Ensure intermediate results are treated as boolean by Polars
        combined_condition = cond1 & cond2 & cond3
        
        df = df.with_columns([
            pl.when(combined_condition)
            .then(((pl.col("Legal Reg Rent") - pl.col("prev_legal_reg_rent")) / pl.col("prev_legal_reg_rent")))
            .otherwise(None)
            .cast(pl.Float64) # Ensure it's float, Nones will be preserved
            .alias("rent_increase_percentage")
        ])
        df = df.drop("prev_legal_reg_rent")
        log.info("Added 'rent_increase_percentage' column.")
    else:
        log.warning("Skipping rent increase calculation: 'Legal Reg Rent' column is missing or not Float64.")
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("rent_increase_percentage"))

    return df

if __name__ == "__main__":
    SAMPLE_CSV_PATH = "apt_sample.csv"
    log.info("Starting vacancy allowance calculation process using new function...")

    try:
        rgb_order_data = load_rgb_orders()
        if rgb_order_data is None or rgb_order_data.is_empty():
            log.error("Failed to load RGBO data. Exiting.")
            exit(1)
        log.info("RGBO data loaded successfully.")

        if not os.path.exists(SAMPLE_CSV_PATH):
            log.error(f"Sample CSV file not found at {SAMPLE_CSV_PATH}. Exiting.")
            exit(1)

        log.info(f"Loading sample unit data from: {SAMPLE_CSV_PATH}")
        # Initial load of the sample CSV. Date parsing will be handled by add_vacancy_allowances_to_df
        # if necessary, but it's good practice to try parsing at load if format is known.
        try:
            initial_df = pl.read_csv(SAMPLE_CSV_PATH, try_parse_dates=False) # Let the function handle complex parsing
            # Attempt to parse known date columns to pl.Date if not already, before passing.
            # This is a bit of pre-emptive parsing if the columns exist and are strings.
            if "Lease Began" in initial_df.columns and initial_df["Lease Began"].dtype == pl.String:
                 initial_df = initial_df.with_columns(pl.col("Lease Began").str.strptime(pl.Date, "%m/%d/%Y", strict=False))
            if "Lease Ends" in initial_df.columns and initial_df["Lease Ends"].dtype == pl.String:
                 initial_df = initial_df.with_columns(pl.col("Lease Ends").str.strptime(pl.Date, "%m/%d/%Y", strict=False))

        except Exception as e:
            log.error(f"Error loading or initially parsing sample CSV {SAMPLE_CSV_PATH}: {e}")
            exit(1)

        if initial_df.is_empty():
            log.info(f"Sample CSV file {SAMPLE_CSV_PATH} is empty. Exiting.")
            exit()
        log.info(f"Sample data loaded. Initial shape: {initial_df.shape}")

        # Use the new function to process the DataFrame
        df_with_allowances = add_vacancy_allowances_to_df(initial_df, rgb_order_data)

        print("\n--- DataFrame with Calculated Vacancy Allowances (processed by new function) ---")
        # Define a list of columns to select for printing, to handle missing columns gracefully
        print_cols = ["Tenant Name", "Lease Began", "Lease Ends", "term_length", 
                      "is_new_tenant", "previous_preferential_rent_has_value", 
                      "tenant_tenure_years", "x_vacancy_allowance"]
        
        # Filter out columns that are not in the DataFrame to avoid errors
        actual_print_cols = [col for col in print_cols if col in df_with_allowances.columns]
        print(df_with_allowances.select(actual_print_cols))
        
        output_csv_path = "apt_sample_with_allowances.csv"
        df_with_allowances.write_csv(output_csv_path)
        log.info(f"Output saved to {output_csv_path}")

    except FileNotFoundError as e:
        log.error(f"Error: {e}")
    except Exception as e:
        log.error(f"An unexpected error occurred in main execution: {e}", exc_info=True)
        import traceback
        traceback.print_exc() 