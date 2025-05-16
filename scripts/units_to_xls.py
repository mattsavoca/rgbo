import polars as pl
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple, Optional, Any
import logging
import xlsxwriter # Import xlsxwriter
from datetime import date # Add this import

# Configure logging for this module
log = logging.getLogger(__name__)
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s] - %(message)s')

# Ensure add_vacancy_allowances_to_df can be imported
# This assumes calculate_vacancy_allowance.py is in the same 'scripts' directory or PYTHONPATH is set up
try:
    from .calculate_vacancy_allowance import add_vacancy_allowances_to_df
except ImportError:
    log.error("Failed to import add_vacancy_allowances_to_df from .calculate_vacancy_allowance. Ensure the file exists and is in the correct path.")
    # Define a dummy if import fails, to allow module loading but operations will fail
    def add_vacancy_allowances_to_df(df, rgb_data):
        raise ImportError("add_vacancy_allowances_to_df could not be imported.")


def _create_lease_year_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates the 'LEASE YEAR' column based on 'Lease Began' or 'Lease Ends'.
    Sorts by it, converts to string, and replaces '1984' with '1984-INIT'.
    """
    df = df.with_columns(
        pl.when(pl.col("Lease Began").is_not_null())
        .then(pl.col("Lease Began").dt.year())
        .otherwise(
            pl.when(pl.col("Lease Ends").is_not_null())
            .then(pl.col("Lease Ends").dt.year() - 1)
            .otherwise(None)
        )
        .cast(pl.Int32, strict=False)
        .alias("LEASE YEAR_temp_int") # Temporary column for sorting
    )

    # Sort by the integer lease year
    df = df.sort("LEASE YEAR_temp_int", nulls_last=True)

    # Convert to string and apply '1984-INIT'
    df = df.with_columns(
        pl.when(pl.col("LEASE YEAR_temp_int") == 1984)
        .then(pl.lit("1984-INIT"))
        .otherwise(pl.col("LEASE YEAR_temp_int").cast(pl.Utf8))
        .alias("LEASE YEAR")
    ).drop("LEASE YEAR_temp_int")
    return df

def _create_incr_legal_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates the '%INCR (LEGAL)' column by combining 'rent_increase_percentage'
    and 'x_vacancy_allowance' into a two-line string.
    Line 1: Actual rent increase percentage.
    Line 2: (0 | [x_vacancy_allowance as % or blank] | 0)
    """

    def format_line1(rent_increase_val: Optional[float]) -> str:
        if rent_increase_val is None or rent_increase_val == 0.0:
            return "0.00%"
        return f"{rent_increase_val * 100:.2f}%"

    def format_line2_value(x_allowance_val: Any) -> str:
        if x_allowance_val is None: # Handles actual None from the list
            return "0.00%"
        try:
            # Attempt to convert to float. This handles cases where x_allowance_val
            # might be an int, float, or a string representation of a number.
            num_val = float(x_allowance_val)
            if num_val == 0.0:
                return "0.00%"
            return f"{num_val * 100:.2f}%"
        except (ValueError, TypeError):
            # If conversion fails, it's a descriptive string or unexpected type.
            # Check if it's an "Indeterminable" type string.
            if isinstance(x_allowance_val, str) and "Indeterminable" in x_allowance_val:
                return ""  # Blank for "Indeterminable"
            # For other non-numeric strings or types that can't be made float, also return blank for safety.
            return "" 

    # Extract columns to Python lists. 
    # Ensure these columns exist as they are expected from add_vacancy_allowances_to_df
    if "rent_increase_percentage" not in df.columns:
        log.error("'rent_increase_percentage' column missing. Cannot create '%INCR (LEGAL)'. Adding as blank.")
        df = df.with_columns(pl.lit("(0 | | 0)").alias("%INCR (LEGAL)")) # Default error string
        return df
    if "x_vacancy_allowance" not in df.columns:
        log.error("'x_vacancy_allowance' column missing. Cannot create '%INCR (LEGAL)'. Adding as blank.")
        df = df.with_columns(pl.lit("(0 | | 0)").alias("%INCR (LEGAL)")) # Default error string
        return df
        
    rent_increase_values = df.get_column("rent_increase_percentage").to_list()
    x_vacancy_values = df.get_column("x_vacancy_allowance").to_list()

    combined_incr_legal_strings = []
    for rip_val, x_val in zip(rent_increase_values, x_vacancy_values):
        line1_str = format_line1(rip_val)
        line2_val_str = format_line2_value(x_val)
        # Construct the two-line string with a newline character
        combined_str = f"{line1_str}\n(0 | {line2_val_str} | 0)"
        combined_incr_legal_strings.append(combined_str)

    df = df.with_columns(
        pl.Series(name="%INCR (LEGAL)", values=combined_incr_legal_strings, dtype=pl.Utf8)
    )
    return df

def _create_pref_rent_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates the 'Pref.' column.
    Displays 'Actual Rent Paid' if it's less than 'Legal Reg Rent', otherwise null.
    """
    df = df.with_columns(
        pl.when(pl.col("Actual Rent Paid") < pl.col("Legal Reg Rent"))
        .then(pl.col("Actual Rent Paid"))
        .otherwise(None) # Represent as null, Polars will handle for Excel
        .cast(pl.Float64, strict=False) # Ensure it can hold floats or nulls
        .alias("Pref.")
    )
    return df

def generate_xlsx_from_units_data(
    processed_data: Dict[str, pl.DataFrame],
    rgb_data: pl.DataFrame
) -> Optional[str]:
    """
    Processes multiple unit DataFrames, applies vacancy calculations, transforms columns,
    and compiles them into a single XLSX file with each unit as a sheet.

    Args:
        processed_data: A dictionary where keys are unit names (sheet names)
                        and values are Polars DataFrames for each unit.
        rgb_data: A Polars DataFrame containing RGBO data for vacancy calculations.

    Returns:
        The path to the generated XLSX file, or None if an error occurs.
    """
    if not processed_data:
        log.warning("No processed data provided to generate XLSX.")
        return None
    if rgb_data is None or rgb_data.is_empty():
        log.warning("RGBO data is missing or empty. Cannot calculate vacancy allowances for XLSX.")
        # Decide if to proceed without vacancy allowances or return None
        # For now, let's make vacancy allowance calculation critical
        return None

    log.info(f"Starting XLSX generation for {len(processed_data)} units.")
    
    processed_dfs_for_excel: Dict[str, pl.DataFrame] = {}

    for unit_name, unit_df_original in processed_data.items():
        log.info(f"Processing unit: {unit_name} for XLSX...")
        if not isinstance(unit_df_original, pl.DataFrame) or unit_df_original.is_empty():
            log.warning(f"Skipping unit '{unit_name}' due to invalid or empty DataFrame.")
            continue
        
        try:
            # 1. Clone the DataFrame to avoid modifying the original in shared state
            df = unit_df_original.clone()

            # 2. Apply vacancy allowance calculations (adds 'x_vacancy_allowance')
            df = add_vacancy_allowances_to_df(df, rgb_data)
            log.debug(f"Unit '{unit_name}': Applied vacancy allowances. Columns: {df.columns}")

            # 3. Create 'LEASE YEAR' column and sort
            df = _create_lease_year_column(df)
            log.debug(f"Unit '{unit_name}': Created and formatted 'LEASE YEAR'.")

            # 4. Create 'ABT.' column (blank)
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("ABT."))
            log.debug(f"Unit '{unit_name}': Added 'ABT.' column.")

            # 5. Create '%INCR (LEGAL)' column
            df = _create_incr_legal_column(df)
            log.debug(f"Unit '{unit_name}': Created '%INCR (LEGAL)' column.")
            
            # 6. Create 'Pref.' column
            df = _create_pref_rent_column(df)
            log.debug(f"Unit '{unit_name}': Created 'Pref.' column.")

            # 7. Ensure all original source columns for aliasing exist, fill with null if not
            required_source_cols = {
                "Apt Status": "APT STAT", "Legal Reg Rent": "REG. RENT",
                "Filing Date": "Date Filed", "Actual Rent Paid": "Act.",
                "Tenant Name": "Tenant Name", "Lease Began": "Lease Began", "Lease Ends": "Lease Ends"
            }
            for source_col in required_source_cols.keys():
                if source_col not in df.columns:
                    log.warning(f"Unit '{unit_name}': Source column '{source_col}' not found. Adding as null.")
                    # Determine appropriate null type, default to Utf8 for safety or match target type
                    if source_col in ["Legal Reg Rent", "Actual Rent Paid"]:
                         df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(source_col))
                    elif source_col in ["Filing Date", "Lease Began", "Lease Ends"]:
                         df = df.with_columns(pl.lit(None).cast(pl.Date).alias(source_col))
                    else: # Apt Status, Tenant Name
                         df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(source_col))


            # 8. Select and alias columns for the final sheet
            # Order of columns as per user request
            final_columns_ordered = [
                pl.col("ABT."),
                pl.col("LEASE YEAR"),
                pl.col("Apt Status").alias("APT STAT"),
                pl.col("Legal Reg Rent").alias("REG. RENT"),
                pl.col("%INCR (LEGAL)"),
                pl.col("Filing Date").alias("Date Filed").cast(pl.Date, strict=False), # Ensure date type if not already
                pl.col("Pref."),
                pl.col("Actual Rent Paid").alias("Act."),
                pl.col("Tenant Name"),
                pl.col("Lease Began").cast(pl.Date, strict=False), # Ensure date type
                pl.col("Lease Ends").cast(pl.Date, strict=False)   # Ensure date type
            ]
            
            df_final_sheet = df.select(final_columns_ordered)
            log.info(f"Unit '{unit_name}': Prepared final sheet. Shape: {df_final_sheet.shape}")
            
            # Sanitize sheet name (Excel has limitations, e.g., length, certain chars)
            # Basic sanitization: replace invalid chars and truncate
            safe_sheet_name = unit_name.replace("/", "-").replace("\\", "-").replace("?", "").replace("*", "").replace("[", "").replace("]", "")
            safe_sheet_name = safe_sheet_name[:30] # Max sheet name length is 31

            processed_dfs_for_excel[safe_sheet_name] = df_final_sheet

        except Exception as e:
            log.error(f"Error processing unit '{unit_name}' for XLSX: {e}", exc_info=True)
            # Optionally, add a sheet with an error message or skip the sheet
            error_df = pl.DataFrame({"Error": [f"Failed to process unit {unit_name}: {str(e)}"]})
            processed_dfs_for_excel[f"ERROR_{unit_name[:25]}"] = error_df


    if not processed_dfs_for_excel:
        log.warning("No dataframes were successfully processed for XLSX export.")
        return None

    # Create a temporary file path for the workbook
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx", prefix="dhcr_combined_") as tmp_file:
            xlsx_path = tmp_file.name
    except Exception as e:
        log.error(f"Failed to create temporary file for XLSX: {e}", exc_info=True)
        return None

    try:
        # Create an XlsxWriter workbook object
        with xlsxwriter.Workbook(xlsx_path) as workbook:
            for sheet_name, df_sheet in processed_dfs_for_excel.items():
                # Add a worksheet
                worksheet = workbook.add_worksheet(sheet_name)
                
                # Write the DataFrame to this worksheet using Polars' integration with an existing workbook/worksheet
                # We pass the workbook and the specific worksheet name (or XlsxWriter worksheet object)
                df_sheet.write_excel(
                    workbook=workbook, 
                    worksheet=worksheet, # Pass the XlsxWriter worksheet object
                    autofit=True,
                    # table_style="Table Style Light 1" # Optional: add table styling
                    # Other formatting options can be applied here too if needed, using xlsxwriter capabilities via Polars
                )
                log.info(f"Added sheet: '{sheet_name}' to '{Path(xlsx_path).name}'")

        log.info(f"Successfully generated XLSX file at: {xlsx_path}")
        return xlsx_path
    except Exception as e:
        log.error(f"Failed to write XLSX file using XlsxWriter: {e}", exc_info=True)
        # Attempt to clean up the temporary file if workbook creation/writing failed
        try:
            if Path(xlsx_path).exists():
                Path(xlsx_path).unlink()
        except Exception as cleanup_err:
            log.error(f"Failed to clean up temp file {xlsx_path} after write error: {cleanup_err}")
        return None

if __name__ == '__main__':
    # Example Usage (for testing this script directly)
    log.info("Running units_to_xls.py directly for testing.")

    # Create dummy RGBO data
    dummy_rgbo_data = pl.DataFrame({
        "order_number": ["55"], "beginning_date": [date(2023,10,1)], "end_date": [date(2024,9,30)],
        "one_year_rate": [0.03], "two_year_rate": [0.0275], "vacancy_lease_rate": [None], # Example values
        # Add other columns expected by add_vacancy_allowances_to_df if necessary
        "one_year_first_half_rate": [None], "one_year_second_half_rate": [None],
        "two_year_first_year_rate": [None], "two_year_second_year_rate": [None],
    }).with_columns([
        pl.col("beginning_date").cast(pl.Date),
        pl.col("end_date").cast(pl.Date)
    ])


    # Create dummy processed_data
    dummy_unit1_data = {
        "Apt Number": ["1A", "1A"],
        "Tenant Name": ["Tenant X", "Tenant Y"],
        "Lease Began": [date(2022, 1, 1), date(2023, 1, 1)],
        "Lease Ends": [date(2022, 12, 31), date(2023, 12, 31)],
        "Legal Reg Rent": [1000.0, 1000.0],
        "Actual Rent Paid": [950.0, 1000.0],
        "Apt Status": ["Registered", "Registered"],
        "Filing Date": [date(2022,1,15), date(2023,1,15)],
        # Minimal columns for add_vacancy_allowances_to_df:
        # 'Lease Began', 'Tenant Name', 'Apt Status', 'Legal Reg Rent', 'Actual Rent Paid'
        # also 'term_length' and 'is_new_tenant' are derived by it.
    }
    dummy_unit2_data = {
        "Apt Number": ["2B"],
        "Tenant Name": ["Tenant Z"],
        "Lease Began": [date(1984, 5, 15)], # Test 1984-INIT
        "Lease Ends": [date(1985, 5, 14)],
        "Legal Reg Rent": [500.0],
        "Actual Rent Paid": [500.0],
        "Apt Status": ["Registered"],
        "Filing Date": [date(1984,6,1)],
        "x_vacancy_allowance": ["Indeterminable: Test"] # Test pre-existing string allowance
    }
    
    dummy_processed_data = {
        "Unit_1A": pl.DataFrame(dummy_unit1_data),
        "Unit_2B": pl.DataFrame(dummy_unit2_data)
    }
    
    # Ensure date columns are of Date type
    for name, df_ in dummy_processed_data.items():
        for col_name in ["Lease Began", "Lease Ends", "Filing Date"]:
            if col_name in df_.columns:
                 dummy_processed_data[name] = df_.with_columns(pl.col(col_name).cast(pl.Date, strict=False))


    output_file = generate_xlsx_from_units_data(dummy_processed_data, dummy_rgbo_data)

    if output_file:
        log.info(f"Test XLSX generated: {output_file}")
    else:
        log.error("Test XLSX generation failed.") 