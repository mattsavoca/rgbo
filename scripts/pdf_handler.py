import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any, Generator
import polars as pl
from datetime import date, datetime
import os # Make sure os is imported if not already

# Attempt to import the necessary function and dependencies from pdf_pipeline
try:
    from .pdf_pipeline import process_pdf # Relative import
    # Removed pdf_to_images import as it's used within process_pdf
    # Assuming pdf_pipeline handles its internal imports (like pdf_to_png)
except ImportError as e:
    logging.error(f"Failed to import from .pdf_pipeline: {e}")
    # Define dummy functions if import fails, to prevent app crash
    def process_pdf(*args, **kwargs):
        logging.error("pdf_pipeline.process_pdf could not be imported.")
        raise RuntimeError("PDF processing unavailable.")
    def pdf_to_images(*args, **kwargs):
        logging.error("pdf_pipeline.pdf_to_images could not be imported.")
        # This might be less critical, so maybe just log the error

# Import vacancy calculation logic
try:
    # Relative import from the same directory now
    from .calculate_vacancy_allowance import load_rgb_orders as primary_load_rgb_orders, add_vacancy_allowances_to_df
except ImportError as e:
    logging.error(f"Failed to import from .calculate_vacancy_allowance: {e}")
    def calculate_vacancy_allowance(*args, **kwargs):
        logging.error("calculate_vacancy_allowance could not be imported.")
        return "Error: Calculation function unavailable"
    def load_rgb_orders(*args, **kwargs):
        logging.error("load_rgb_orders could not be imported.")
        raise RuntimeError("RGBO data loading unavailable.")

# Configure logging for the handler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [PDF_HANDLER] - %(message)s')

# --- Load RGBO Data ---
# Get the directory where this script (pdf_handler.py) resides
HANDLER_DIR = Path(__file__).parent.resolve() # This will now be the 'scripts' directory
# Construct the path to rgbo.csv relative to the PARENT of this script's directory
# Assumes rgbo.csv is in the root directory (one level up from 'scripts')
RGBO_CSV_FULL_PATH = HANDLER_DIR.parent / "rgbo.csv"

# Load this once when the module is loaded.
# Consider potential issues if the app runs long and the CSV changes.
RGBO_DATA: Optional[pl.DataFrame] = None # Initialize to None
try:
    logging.info(f"Attempting to load RGBO data from: {RGBO_CSV_FULL_PATH}")
    if not RGBO_CSV_FULL_PATH.is_file():
        raise FileNotFoundError(f"RGBO file not found at derived path: {RGBO_CSV_FULL_PATH}")

    # Pass the explicit full path to the loading function
    RGBO_DATA = primary_load_rgb_orders(csv_path=RGBO_CSV_FULL_PATH)

    if RGBO_DATA is None or RGBO_DATA.is_empty():
         logging.warning("Loaded RGBO data is empty or None after processing. Vacancy calculations may fail.")
    else:
         logging.info(f"RGBO data loaded successfully ({len(RGBO_DATA)} records) from {RGBO_CSV_FULL_PATH} for vacancy calculations.")
except FileNotFoundError as e:
    logging.error(f"{e} Vacancy calculations will likely fail.")
    # Allow app to continue, but RGBO_DATA remains None
except Exception as e:
    logging.error(f"Failed to load or process RGBO data from {RGBO_CSV_FULL_PATH}: {e}", exc_info=True)
    # Allow app to continue, RGBO_DATA remains None

# Define a type alias for the final result of run_pdf_processing
ProcessingResult = Tuple[List[Dict[str, Any]], Optional[Path]]
ErrorResult = Tuple[str, str] # For ("ERROR", error_message)
# Update the generator's yield type to include string status or the final result/error
YieldType = Union[str, Tuple[str, Union[ProcessingResult, str]]]

def run_pdf_processing(
    pdf_path: Path, 
    base_output_dir: Path, 
    generate_images: bool, 
    calculate_vacancy: bool
) -> Generator[YieldType, None, None]: # Changed return type annotation for generator
    """
    Runs the PDF processing pipeline for a given PDF file and calculates vacancy allowances.
    This function is a generator. It yields status strings during processing.
    As its final act, it yields a tuple: 
    - ("SUCCESS", (pipeline_results_list, run_output_dir_relative_path)) on success
    - ("ERROR", error_message_string) on critical failure before completion.
    
    Args:
        pdf_path: Path to the input PDF file.
        base_output_dir: The base directory where output should be stored (e.g., app's UPLOAD_DIR).
        generate_images: Flag indicating whether to generate images.
        calculate_vacancy: Flag indicating whether to perform vacancy allowance calculation.
        
    Returns:
        A generator that yields status strings and results.
    """
    if not pdf_path.exists():
        logging.error(f"Input PDF not found: {pdf_path}")
        # For a generator, we yield the error then return to stop iteration.
        yield ("ERROR", f"Input PDF not found: {pdf_path}")
        return

    # Create a unique subdirectory for this specific PDF's output
    run_output_dir_name_str = pdf_path.stem
    run_output_dir = base_output_dir / run_output_dir_name_str
    try:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        yield f"[Handler] Created output directory: {run_output_dir}"
    except Exception as e:
        logging.error(f"[Handler] Failed to create output directory {run_output_dir}: {e}")
        yield ("ERROR", f"Failed to create output directory: {e}")
        return

    try:
        yield f"[Handler] Starting PDF processing for {pdf_path.name}..."
        # process_pdf is now a generator from pdf_pipeline.py
        # Yield its status messages directly
        yield from process_pdf(pdf_path, run_output_dir, generate_images)
        yield f"[Handler] Core PDF processing pipeline finished for {pdf_path.name}."

    except Exception as e:
        logging.error(f"[Handler] Error during core pdf_pipeline.process_pdf execution: {e}", exc_info=True)
        yield ("ERROR", f"Error during core PDF processing: {e}")
        return # Stop if core processing fails critically

    # --- Calculate Vacancy Allowances ---
    run_output_dir_name_str = pdf_path.stem # Ensure this is defined before use
    run_output_dir = base_output_dir / run_output_dir_name_str # Reconstruct full path if needed

    if calculate_vacancy:
        yield "[Handler] Vacancy calculation requested."
        if RGBO_DATA is None or RGBO_DATA.is_empty():
            yield "[Handler] WARNING: Skipping vacancy allowance calculation - RGBO data unavailable."
        else:
            yield "[Handler] Calculating vacancy allowances for generated unit CSVs..."
            # Ensure run_output_dir exists before globbing
            if not run_output_dir.is_dir():
                yield f"[Handler] WARNING: Output directory {run_output_dir} not found. Cannot calculate vacancies."
            else:
                csv_files_for_vacancy = list(run_output_dir.glob('apt_*.csv'))
                if not csv_files_for_vacancy:
                    yield "[Handler] No CSV files found to calculate vacancy allowances for."
                else:
                    for i, csv_file in enumerate(csv_files_for_vacancy):
                        yield f"[Handler] Vacancy calc for {csv_file.name} ({i+1}/{len(csv_files_for_vacancy)})..."
                        try:
                            # Load CSV
                            df = pl.read_csv(csv_file, try_parse_dates=True)

                            if df.is_empty():
                                 logging.warning(f"CSV file {csv_file.name} is empty. Skipping vacancy calculation.")
                                 # Optionally ensure 'Vacancy Allowance' column doesn't exist if needed
                                 # Or add it as None if downstream code expects it?
                                 # For now, just skip processing.
                                 continue

                            # --- Feature Engineering (using Polars expressions) ---
                            required_source_cols = ["Lease Began", "Lease Ends", "Legal Reg Rent", "Actual Rent Paid", "Apt Status", "Tenant Name"]
                            missing_cols = [col for col in required_source_cols if col not in df.columns]
                            if missing_cols:
                                 logging.warning(f"Skipping vacancy calc for {csv_file.name}: Missing required source columns: {missing_cols}.")
                                 # Decide how to handle this: Add null column or just skip?
                                 # Adding null for consistency if downstream expects the column.
                                 try:
                                      if "Vacancy Allowance" not in df.columns:
                                           df = df.with_columns(pl.lit(None).cast(pl.Object).alias("Vacancy Allowance"))
                                           df.write_csv(csv_file)
                                 except Exception as write_err:
                                      logging.error(f"Could not add placeholder Vacancy Allowance column to {csv_file.name} after missing source columns: {write_err}")
                                 continue

                            # Ensure correct types after load
                            df = df.with_columns([
                                pl.col("Lease Began").cast(pl.Date, strict=False),
                                pl.col("Lease Ends").cast(pl.Date, strict=False),
                                pl.col("Legal Reg Rent").cast(pl.Float64, strict=False),
                                pl.col("Actual Rent Paid").cast(pl.Float64, strict=False),
                                pl.col("Apt Status").cast(pl.String, strict=False).fill_null(""),
                                pl.col("Tenant Name").cast(pl.String, strict=False).fill_null("")
                            ])

                            # Sort by lease start date
                            df = df.sort("Lease Began", nulls_last=True)

                            # --- Feature Engineering expressions ---
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
                                pl.lit(False).alias("had_vacancy_allowance_in_prev_12_mo"),
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

                            # --- Tenant Tenure Calculation (using window function) ---
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

                            # --- Define the function to apply for vacancy calculation ---
                            def apply_vacancy_calc(row_struct: dict) -> Union[float, str, None]:
                                # No need to check RGBO_DATA here again, already checked outside loop
                                unit_data = {
                                    'apartment_status': row_struct.get("Apt Status"),
                                    'is_new_tenant': row_struct.get("is_new_tenant"),
                                    'term_length': row_struct.get("term_length"),
                                    'lease_start_date': row_struct.get("lease_start_date"),
                                    'had_vacancy_allowance_in_prev_12_mo': row_struct.get("had_vacancy_allowance_in_prev_12_mo"),
                                    'previous_preferential_rent_has_value': row_struct.get("previous_preferential_rent_has_value"),
                                    'tenant_tenure_years': row_struct.get("tenant_tenure_years")
                                }

                                if unit_data['lease_start_date'] is None:
                                    return "Indeterminable: Missing Lease Start Date"
                                if unit_data['apartment_status'] is None:
                                     unit_data['apartment_status'] = 'Unknown'

                                try:
                                    result = calculate_vacancy_allowance(unit_data, RGBO_DATA)
                                    return result
                                except Exception as e:
                                    logging.error(f"Error applying vacancy calculation logic for row data {unit_data}: {e}", exc_info=True)
                                    return f"Error: Calculation Failed"

                            # --- Apply the calculation function row-wise ---
                            required_cols_for_calc = [
                                "Apt Status", "is_new_tenant", "term_length", "lease_start_date",
                                "had_vacancy_allowance_in_prev_12_mo", "previous_preferential_rent_has_value",
                                "tenant_tenure_years"
                            ]

                            struct_cols = [c for c in required_cols_for_calc if c in df.columns]
                            if len(struct_cols) != len(required_cols_for_calc):
                                missing_for_struct = set(required_cols_for_calc) - set(struct_cols)
                                logging.error(f"Cannot perform vacancy calculation for {csv_file.name}: Columns needed for logic missing: {missing_for_struct}. Adding null column.")
                                if "Vacancy Allowance" not in df.columns:
                                     df = df.with_columns(pl.lit(None).cast(pl.Object).alias("Vacancy Allowance"))
                            else:
                                 df = df.with_columns(
                                     pl.struct(struct_cols)
                                       .map_elements(
                                           apply_vacancy_calc,
                                           return_dtype=pl.Object
                                       )
                                       .alias("Vacancy Allowance")
                                 )

                            # Save the updated DataFrame back to CSV
                            df.write_csv(csv_file)
                            logging.info(f"Successfully calculated vacancy allowance and updated: {csv_file.name}")

                        except pl.exceptions.ColumnNotFoundError as e:
                             logging.error(f"Column not found error processing vacancy for {csv_file.name}: {e}. Skipping this file.")
                        except Exception as e:
                            logging.error(f"Failed to calculate vacancy allowance for {csv_file.name}: {e}", exc_info=True)
                            try:
                                if 'df' in locals() and isinstance(df, pl.DataFrame) and "Vacancy Allowance" not in df.columns:
                                     df = df.with_columns(pl.lit(f"Error: Processing Failed").cast(pl.Object).alias("Vacancy Allowance"))
                                     df.write_csv(csv_file)
                                     logging.info(f"Added error placeholder to Vacancy Allowance column for {csv_file.name}")
                            except Exception as E2:
                                 logging.error(f"Could not even write error placeholder to {csv_file.name}: {E2}")
    else:
         yield "[Handler] Vacancy calculation was not requested. Skipping."

    # --- Scan for results (This part builds the 'results' list for the final yield) ---
    yield "[Handler] Scanning for processing results..."
    results_list: List[Dict[str, Any]] = [] # Renamed to avoid conflict with 'results' variable name
    apt_files_map: Dict[str, Dict[str, Any]] = {} # Renamed to avoid conflict
    image_dir_path: Optional[Path] = None # Renamed
    if generate_images:
        image_dir_path = run_output_dir / f"{pdf_path.stem}_images"

    # Scan for CSV files and try to extract unit name
    if run_output_dir.is_dir(): # Check if dir exists before globbing
        for csv_file in run_output_dir.glob('apt_*.csv'):
            match = re.match(r"apt_(.+)\.csv", csv_file.name)
            unit_name = match.group(1).replace('_', '/') if match else csv_file.stem
            # Paths should be relative to base_output_dir, so run_output_dir_name_str / csv_file.name
            relative_csv_path = Path(run_output_dir_name_str) / csv_file.name
            apt_files_map[unit_name] = {
                'unit_name': unit_name,
                'csv_path': relative_csv_path,
                'img_paths': []
            }
            logging.debug(f"[Handler] Found CSV for unit '{unit_name}': {relative_csv_path}")
    else:
        yield f"[Handler] WARNING: Output directory {run_output_dir} does not exist. Cannot scan for CSV results."

    # Scan for image files if generated
    if image_dir_path and image_dir_path.exists():
        for img_file in image_dir_path.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # relative_img_path should be base_output_dir / run_output_dir_name_str / image_dir.name / img_file.name
                relative_img_path = Path(run_output_dir_name_str) / image_dir_path.name / img_file.name
                logging.debug(f"[Handler] Found image: {relative_img_path}")
                for unit_data in apt_files_map.values():
                    if relative_img_path not in unit_data['img_paths']:
                        unit_data['img_paths'].append(relative_img_path)

    results_list = list(apt_files_map.values())

    if not results_list and run_output_dir.is_dir() and any(run_output_dir.iterdir()):
        # This means output files (e.g. images) might exist, but no CSVs were parsed into results_list.
        yield f"[Handler] Processing seems to have produced some output files in {run_output_dir}, but no specific unit CSV data was compiled."
    elif not results_list:
        yield f"[Handler] WARNING: No apartment CSV files or other outputs found in {run_output_dir} after processing."
    else:
        yield f"[Handler] Successfully compiled {len(results_list)} unit result(s)."

    # Final yield: a tuple indicating success and containing the data
    # The relative path to the specific output directory for this run
    final_run_output_dir_relative_path = Path(run_output_dir_name_str)
    yield ("SUCCESS", (results_list, final_run_output_dir_relative_path))

