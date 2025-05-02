import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import polars as pl
import re # Added for sanitizing docket number if needed again

# Attempt to import the CBB processing logic
try:
    from .cbb_pipeline import process_cbb_pdf
    # pdf_to_images is handled within process_cbb_pdf now
except ImportError as e:
    logging.error(f"Failed to import from .cbb_pipeline: {e}", exc_info=True)
    # Define dummy function if import fails
    def process_cbb_pdf(*args, **kwargs) -> List:
        logging.error("cbb_pipeline.process_cbb_pdf could not be imported.")
        raise RuntimeError("CBB PDF processing unavailable due to import error.")

# Configure logging for the handler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [CBB_HANDLER] - %(message)s')

def run_cbb_processing(pdf_path: Path, base_output_dir: Path, generate_images: bool) -> Tuple[List[Dict[str, Union[str, Path]]], Optional[Path]]:
    """
    Runs the CBB PDF processing pipeline for a given PDF file.

    Args:
        pdf_path: Path to the input CBB PDF file.
        base_output_dir: The base directory where output should be stored (e.g., app's UPLOAD_DIR).
        generate_images: Flag indicating whether to generate images.

    Returns:
        A tuple containing:
        - A list of dictionaries, each representing a docket with its number
          and the RELATIVE path to its output CSV file.
          Example: [{'docket_no': 'LD123456OR', 'csv_path': Path('cbb_report_stem/docket_LD123456OR.csv')}]
        - The relative path to the specific output directory for this run, or None if processing failed early.
    """
    if not pdf_path.exists():
        logging.error(f"Input CBB PDF not found: {pdf_path}")
        return [], None

    # Create a unique subdirectory for this specific PDF's output
    run_output_dir_name = pdf_path.stem # e.g., "cbb_report_name"
    run_output_dir = base_output_dir / run_output_dir_name # e.g., /uploads/cbb_report_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created CBB output directory for this run: {run_output_dir}")

    results_list: List[Dict[str, Union[str, Path]]] = []
    relative_run_output_path = Path(run_output_dir_name) # Relative path for return

    try:
        # Ensure environment (API keys) is ready (handled by cbb_pipeline loading .env)
        logging.info(f"Starting CBB PDF processing for {pdf_path}...")
        # process_cbb_pdf now handles CSV creation and returns info needed
        # It expects the full path to the specific output directory for this run.
        exported_files_info = process_cbb_pdf(pdf_path, run_output_dir, generate_images=generate_images)
        logging.info(f"CBB PDF processing finished for {pdf_path}.")

        # --- Structure results for Gradio --- #
        if exported_files_info:
             for file_info in exported_files_info:
                 docket_no = file_info.get('docket_no')
                 csv_filename = file_info.get('csv_filename')
                 if docket_no and csv_filename:
                     # Construct the RELATIVE path to the CSV from the base_output_dir perspective
                     relative_csv_path = relative_run_output_path / csv_filename
                     results_list.append({
                         'docket_no': docket_no,
                         'csv_path': relative_csv_path
                     })
                 else:
                     logging.warning(f"Skipping result entry due to missing docket_no or csv_filename: {file_info}")
        else:
             logging.warning(f"process_cbb_pdf completed but returned no exported file information for {pdf_path.name}. Output dir: {run_output_dir}")

        # Also check if images were generated and add their paths if needed by UI (currently not required by spec)
        if generate_images:
            image_dir = run_output_dir / f"{pdf_path.stem}_images"
            if image_dir.exists():
                # If UI needed image paths per docket, we'd need to associate them here.
                # For now, just log their existence.
                logging.info(f"Image directory exists at: {image_dir}")

    except RuntimeError as rterr:
         logging.error(f"RuntimeError during CBB processing (likely import issue): {rterr}")
         return [], relative_run_output_path # Return empty results but provide the directory path
    except Exception as e:
        logging.error(f"Error during cbb_pipeline.process_cbb_pdf execution: {e}", exc_info=True)
        # Return empty results but provide the directory path for potential cleanup/zip
        return [], relative_run_output_path

    if not results_list and not list(run_output_dir.glob('*')):
         logging.warning(f"No docket CSV files found and no other files (like images) found in {run_output_dir} after processing. Check CBB PDF content and processing logs.")
    elif not results_list:
         logging.warning(f"No docket CSV files were generated or reported by the pipeline in {run_output_dir}, although other files might exist.")

    # Return the list of results (docket_no -> relative csv path) and the relative path to the run directory
    return results_list, relative_run_output_path

# Example usage (if needed for standalone testing)
if __name__ == '__main__':
    print("This script is intended to be imported as a module.")
    # Add test code here if desired, e.g.:
    # test_pdf = Path("path/to/your/test_cbb.pdf")
    # test_output = Path("./test_cbb_output")
    # if test_pdf.exists():
    #     results, out_dir = run_cbb_processing(test_pdf, test_output, generate_images=True)
    #     print("Results:", results)
    #     print("Output Dir Relative Path:", out_dir)
    # else:
    #     print(f"Test PDF not found at {test_pdf}") 