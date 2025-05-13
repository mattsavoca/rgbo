import os
import requests
import json
import re
import csv
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Optional, Union, Dict, Tuple
from datetime import date, datetime
from google.api_core import exceptions as google_exceptions
from google import genai
from google.genai import Client, types
from collections import defaultdict
import logging
import time
from dotenv import load_dotenv
import argparse
# Relative import from the same directory
try:
    from .pdf_to_png import pdf_to_images
except ImportError:
    logging.error("Failed to import pdf_to_images. Image generation will fail.")
    def pdf_to_images(*args, **kwargs):
        raise ImportError("pdf_to_images function is unavailable.")

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [CBB_PIPELINE] - %(message)s')

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash-latest") # Default model if not set
LEN_TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT_MS", 120000)) # Default timeout 2 mins
DEFAULT_OUTPUT_DIR = Path("./cbb_output_csvs")

# --- Pydantic Models ---

# Using the flexible date parser from the DHCR pipeline
def parse_flexible_date(value: Optional[Union[str, date]]) -> Optional[date]:
    """Attempts to parse various date formats, returning None if parsing fails."""
    if isinstance(value, date):
        return value
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if value.lower() in ["", "none", "n/a", "na"] or value.upper() == "NC":
        return None

    m_d_y_pattern = re.compile(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2})$")
    match = m_d_y_pattern.match(value)
    if match:
        month, day, year = match.groups()
        yr_str = year
        yr_int = int(yr_str)
        cent_prefix = "19" if yr_int > 50 else "20"
        new_yr = cent_prefix + yr_str
        try:
            return datetime.strptime(f"{month}/{day}/{new_yr}", "%m/%d/%Y").date()
        except (ValueError, TypeError):
            pass

    formats_to_try = [
        "%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d-%b-%Y", "%B %d, %Y", "%Y%m%d",
        "%m/%d/%y", # Added format MM/DD/YY
    ]
    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(value.split(' ')[0], fmt)
            # Handle two-digit years common in CBB format
            if fmt == "%m/%d/%y":
                if dt.year > datetime.now().year + 10: # Heuristic: if year > future+10, assume 19xx
                    dt = dt.replace(year=dt.year - 100)
            return dt.date()
        except (ValueError, TypeError):
            continue
    logging.warning(f"Could not parse date: {value}")
    return None

class CBB_Entry(BaseModel):
    bldg_id: Optional[str] = Field(None, alias="Bldg Id", description="Building ID from page header")
    address: Optional[str] = Field(None, alias="Address", description="Building Address from page header")
    docket_no: str = Field(..., alias="Docket No", description="Case Docket Number")
    apt_no: Optional[str] = Field(None, alias="Apt No", description="Apartment Number")
    file_date: Optional[date] = Field(None, alias="File Date", description="Date the case was filed")
    start_date: Optional[date] = Field(None, alias="Start Date", description="Start date associated with the case")
    status_disp: Optional[str] = Field(None, alias="Status/Disp", description="Status or Disposition of the case")
    case_type: Optional[str] = Field(None, alias="Case Type", description="Type of the case")

    @field_validator('file_date', 'start_date', mode='before')
    @classmethod
    def parse_dates(cls, value):
        return parse_flexible_date(value)

    @field_validator('bldg_id', 'address', 'docket_no', 'apt_no', 'status_disp', 'case_type', mode='before')
    @classmethod
    def clean_strings(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value

    # Provide aliases for CSV export consistency
    def to_dict_for_csv(self):
         return {
            "Bldg Id": self.bldg_id,
            "Address": self.address,
            "Docket No": self.docket_no,
            "Apt No": self.apt_no,
            "File Date": self.file_date.isoformat() if self.file_date else None,
            "Start Date": self.start_date.isoformat() if self.start_date else None,
            "Status/Disp": self.status_disp,
            "Case Type": self.case_type,
        }

class CBB_Page_Data(BaseModel):
    bldg_id: Optional[str] = Field(None, description="Building ID extracted from the page header")
    address: Optional[str] = Field(None, description="Address extracted from the page header")
    cbb_entries: List[CBB_Entry] = Field(..., description="List of case entries extracted from the table")


# --- Gemini Interaction Functions ---

# Re-use get_gemini_client from pdf_pipeline (or copy it here if preferred)
# Assuming it's available or copied:
def get_gemini_client():
    """Initializes and returns the Gemini client by explicitly passing the API key."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logging.error("Failed to get GEMINI_API_KEY from environment for client initialization.")
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        client = Client(api_key=api_key)
        logging.info("Gemini client initialized successfully using provided API key.")
        return client
    except ValueError as ve:
        logging.error(f"Client Initialization Error: {ve}")
        raise
    except Exception as e:
        logging.error(f"Failed to initialize Gemini client: {e}")
        raise


def identify_cbb_data_pages(client: Client, pdf_data: bytes) -> List[int]:
    """
    Uses Gemini to identify CBB pages containing the main data table, excluding the title page.

    Args:
        client: The initialized Gemini API Client.
        pdf_data: Raw bytes of the PDF file.

    Returns:
        A list of 1-based page numbers containing valid CBB data tables.
    """
    prompt = (
        "Analyze the provided PDF document, which is a NYS DHCR 'Cases by Building Report' (CBB). "
        "Identify the 1-based page numbers that contain the main data table with columns like 'Docket No', 'Apt No', 'File Date', 'Start Date', 'Status/Disp', 'Case Type'. "
        "These data pages typically have a header containing 'Bldg Id' and 'Address'. "
        "Critically, **exclude** the initial 'Title Page' which describes the report but contains no tabular case data. "
        "Return the result ONLY as a JSON list of integer page numbers, e.g., [2, 3, 4]. "
        "If no pages with the main data table are found, return an empty list []."
    )
    max_retries = 3
    retry_delay = 5 # seconds

    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries}: Identifying CBB data pages...")
            pdf_part = types.Part.from_bytes(data=pdf_data, mime_type="application/pdf")
            logging.info(f"PDF data part prepared. Making generate_content call for CBB page ID...")

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[pdf_part, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=LEN_TIMEOUT)
                ),
            )

            logging.debug(f"Raw CBB page identification response: {response.text}")
            page_numbers = json.loads(response.text)

            if isinstance(page_numbers, list) and all(isinstance(p, int) for p in page_numbers):
                valid_pages = sorted([p for p in page_numbers if p > 0])
                logging.info(f"Identified valid CBB data pages: {valid_pages}")
                return valid_pages
            else:
                logging.warning(f"Received unexpected format for CBB page numbers: {page_numbers}. Retrying...")

        except json.JSONDecodeError as e:
            logging.warning(f"Attempt {attempt + 1}: Failed to decode JSON from CBB page ID response: {e}. Response text: {response.text}")
        except (google_exceptions.DeadlineExceeded, google_exceptions.ServiceUnavailable, requests.exceptions.ReadTimeout) as e:
             logging.warning(f"Attempt {attempt + 1}: Network or timeout error during CBB page ID: {e}. Retrying in {retry_delay}s...")
             time.sleep(retry_delay)
        except google_exceptions.ResourceExhausted as e:
             logging.error(f"Gemini API quota exceeded during CBB page ID: {e}")
             return []
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: Unexpected error during CBB page ID: {e}", exc_info=True)
            time.sleep(retry_delay)

    logging.error("Failed to identify CBB data pages after multiple retries.")
    return []


def extract_data_from_cbb_page(client: Client, pdf_data: bytes, page_number: int) -> Optional[CBB_Page_Data]:
    """
    Uses Gemini to extract the header ('Bldg Id', 'Address') and tabular CBB data
    from a specific PDF page.

    Args:
        client: The initialized Gemini API Client.
        pdf_data: Raw bytes of the PDF file.
        page_number: The 1-based page number to extract data from.

    Returns:
        A CBB_Page_Data object containing the extracted header and entries, or None if extraction fails.
    """
    schema = CBB_Entry.model_json_schema()
    page_data_schema = CBB_Page_Data.model_json_schema()

    prompt = (
        f"From the provided PDF document (a NYS DHCR CBB Report), focus **only** on page {page_number}. "
        "Perform the following two extraction tasks:\n"
        "1. Extract the 'Bldg Id' and 'Address' values, which are typically located in the header area of the page.\n"
        "2. Extract all rows from the main data table present on this page. The table columns are usually: 'Docket No', 'Apt No', 'File Date', 'Start Date', 'Status/Disp', 'Case Type'.\n"
        "Return the extracted data as a single JSON object conforming to the following structure:\n"
        f"{json.dumps(page_data_schema, indent=2)}\n"
        "Where 'bldg_id' and 'address' contain the values from the header, and 'cbb_entries' is a list of objects, each representing one row from the table and conforming to this item schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "Important Parsing Instructions:\n"
        "- Ensure dates ('File Date', 'Start Date') are parsed into 'YYYY-MM-DD' format. Handle formats like 'MM/DD/YY' (infer century: assume 20xx unless > current_year+10, then 19xx). If a date is missing, represent it as null.\n"
        "- Carefully align values to the correct columns, especially accounting for empty cells or slightly misaligned text.\n"
        "- Normalize apartment numbers if possible (e.g., '1A').\n"
        "- If the page does not contain the expected header or table structure, or if the table is empty, return {\"bldg_id\": null, \"address\": null, \"cbb_entries\": []}.\n"
    )


    max_retries = 3
    retry_delay = 5 # seconds

    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries}: Extracting CBB data from page {page_number}...")
            pdf_part = types.Part.from_bytes(data=pdf_data, mime_type="application/pdf")

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[pdf_part, prompt],
                 config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=LEN_TIMEOUT)
                ),
            )

            logging.debug(f"Raw CBB data extraction response for page {page_number}: {response.text}")
            raw_data = json.loads(response.text)

            # Use Pydantic to parse and validate the full structure
            page_data = CBB_Page_Data(**raw_data)

            # --- Post-processing: Inject header data into each entry ---
            # This is crucial as the model returns header and entries separately.
            if page_data.cbb_entries:
                 for entry in page_data.cbb_entries:
                     entry.bldg_id = page_data.bldg_id
                     entry.address = page_data.address
                 logging.info(f"Successfully extracted and validated {len(page_data.cbb_entries)} CBB entries from page {page_number}. Injected header info.")
            else:
                 logging.info(f"Successfully parsed page {page_number}, but no CBB entries found in the table.")

            return page_data # Return the object with header info now inside each entry

        except json.JSONDecodeError as e:
            logging.warning(f"Attempt {attempt + 1}: Failed to decode JSON from CBB page {page_number} response: {e}. Response text: {response.text}")
        except ValidationError as e:
            logging.warning(f"Attempt {attempt + 1}: Pydantic validation failed for CBB page {page_number}: {e}. Data: {response.text}")
        except (google_exceptions.DeadlineExceeded, google_exceptions.ServiceUnavailable, requests.exceptions.ReadTimeout) as e:
             logging.warning(f"Attempt {attempt + 1}: Network/timeout error extracting CBB page {page_number}: {e}. Retrying...")
             time.sleep(retry_delay)
        except google_exceptions.ResourceExhausted as e:
             logging.error(f"Gemini API quota exceeded extracting CBB page {page_number}: {e}")
             return None
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: Unexpected error extracting CBB page {page_number}: {e}", exc_info=True)
            time.sleep(retry_delay)

    logging.error(f"Failed to extract CBB data from page {page_number} after multiple retries.")
    return None


# --- Data Processing and Export ---

def export_docket_to_csv(docket_no: str, docket_entries: List[CBB_Entry], output_dir: Path):
    """Exports the data for a single docket number to a CSV file."""
    if not docket_entries:
        logging.warning(f"No data found for Docket No: {docket_no}. Skipping CSV export.")
        return

    # Sort by File Date (earliest first), then maybe Start Date?
    docket_entries.sort(key=lambda x: (x.file_date or date.min, x.start_date or date.min))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize docket number for filename (replace slashes, etc.)
    safe_docket_no = re.sub(r'[\\/*?:"<>|]', "_", docket_no)
    filename = output_dir / f"docket_{safe_docket_no}.csv"

    logging.info(f"Exporting data for Docket {docket_no} to {filename}")

    headers = list(docket_entries[0].to_dict_for_csv().keys())

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for entry in docket_entries:
                 writer.writerow(entry.to_dict_for_csv())
    except IOError as e:
        logging.error(f"Failed to write CSV file for Docket {docket_no}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during CSV export for Docket {docket_no}: {e}", exc_info=True)


def export_all_entries_to_csv(all_entries: List[CBB_Entry], output_dir: Path, filename: str = "all_dockets_report.csv"):
    """Exports all CBB entries to a single CSV file."""
    if not all_entries:
        logging.warning("No entries found to export to the combined CSV. Skipping.")
        return None

    # Sort entries for consistency, e.g., by Bldg Id, then Docket No, then File Date
    # This is a more comprehensive sort for the combined file.
    all_entries.sort(key=lambda x: (
        x.bldg_id or "",
        x.docket_no or "",
        x.file_date or date.min,
        x.start_date or date.min
    ))

    output_file_path = output_dir / filename
    logging.info(f"Exporting all {len(all_entries)} entries to {output_file_path}")

    # Use the first entry to determine headers, assuming all entries have the same structure.
    headers = list(all_entries[0].to_dict_for_csv().keys())

    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for entry in all_entries:
                writer.writerow(entry.to_dict_for_csv())
        logging.info(f"Successfully exported all entries to {output_file_path}")
        return filename # Return the name of the created file
    except IOError as e:
        logging.error(f"Failed to write combined CSV file {output_file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during combined CSV export to {output_file_path}: {e}", exc_info=True)
        return None


# --- Main Execution Logic ---

def process_cbb_pdf(pdf_filepath: Union[str, Path], output_dir: Path, generate_images: bool = False):
    """Main function to process a single CBB PDF."""
    pdf_path = Path(pdf_filepath)
    if not pdf_path.is_file():
        logging.error(f"CBB PDF file not found: {pdf_path}")
        return [] # Return empty list on file not found

    logging.info(f"Processing CBB PDF: {pdf_path}")

    # 1. Load PDF data
    try:
        logging.info("Reading CBB PDF file into memory...")
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        logging.info(f"Loaded CBB PDF data ({len(pdf_data)} bytes).")
    except IOError as e:
        logging.error(f"Failed to read CBB PDF file {pdf_path}: {e}")
        return []

    # 2. Initialize Gemini Client
    try:
        logging.info("Initializing Gemini client for CBB processing...")
        client = get_gemini_client()
        logging.info("Gemini client initialized.")
    except Exception:
        return [] # Error logged in get_gemini_client

    # 3. Identify relevant pages (skip title page)
    logging.info("Identifying CBB data pages...")
    data_pages = identify_cbb_data_pages(client, pdf_data)
    logging.info(f"Identified CBB data pages: {data_pages}")
    if not data_pages:
        logging.warning(f"No CBB data pages identified in {pdf_path}. Cannot extract data.")
        # Still attempt image generation if requested, but return no CSV data
        if generate_images:
            # (Image generation logic moved to the end, after checking if output dir exists)
            pass # Placeholder
        return []

    # 4. Extract data from each relevant page
    all_cbb_entries: List[CBB_Entry] = []
    page_processing_results = {} # Store per-page results for logging/debugging

    for page_num in data_pages:
        page_data = extract_data_from_cbb_page(client, pdf_data, page_num)
        page_processing_results[page_num] = page_data # Store result object

        if page_data and page_data.cbb_entries:
            # Header info (bldg_id, address) is now injected into each entry by extract_data_from_cbb_page
            all_cbb_entries.extend(page_data.cbb_entries)
            logging.info(f"Added {len(page_data.cbb_entries)} entries from CBB page {page_num}.")
        else:
            logging.warning(f"Could not extract valid CBB entries from page {page_num} (or page had no entries).")


    # --- Log summary of page extraction ---
    successful_pages = [p for p, res in page_processing_results.items() if res and res.cbb_entries]
    empty_pages = [p for p, res in page_processing_results.items() if res and not res.cbb_entries]
    failed_pages = [p for p, res in page_processing_results.items() if not res]
    logging.info(f"CBB Extraction Summary: Success: {len(successful_pages)} pages, Empty: {len(empty_pages)} pages, Failed: {len(failed_pages)} pages.")

    if not all_cbb_entries:
        logging.warning("No CBB entries were successfully extracted from any identified page.")
         # Still attempt image generation if requested
        if generate_images:
             image_output_dir = output_dir / f"{pdf_path.stem}_images"
             logging.info(f"No CSV data, but attempting image generation to: {image_output_dir}")
             try:
                 image_output_dir.mkdir(parents=True, exist_ok=True)
                 pdf_to_images(str(pdf_path), str(image_output_dir))
                 logging.info(f"Successfully converted CBB PDF pages to images in {image_output_dir}")
             except Exception as e:
                 logging.error(f"Failed to convert CBB PDF to images: {e}", exc_info=True)
        return []

    logging.info(f"Total extracted CBB entries across all pages: {len(all_cbb_entries)}")

    # 5. Group data by Docket Number
    grouped_by_docket: Dict[str, List[CBB_Entry]] = defaultdict(list)
    processed_docket_nos = set()
    for entry in all_cbb_entries:
        # Use docket_no as the key
        if entry.docket_no and isinstance(entry.docket_no, str):
             grouped_by_docket[entry.docket_no].append(entry)
             processed_docket_nos.add(entry.docket_no)
        else:
             logging.warning(f"Skipping CBB entry with invalid or missing docket_no: {entry}")

    # 6. Export each docket's data to CSV
    logging.info(f"Found {len(grouped_by_docket)} unique docket numbers. Exporting to CSV in {output_dir}...")
    exported_files_info = []
    for docket_no, docket_entries in grouped_by_docket.items():
        export_docket_to_csv(docket_no, docket_entries, output_dir)
        # Store info needed by the handler/UI
        safe_docket_no = re.sub(r'[\\/*?:"<>|]', "_", docket_no)
        csv_filename = f"docket_{safe_docket_no}.csv"
        exported_files_info.append({
            'docket_no': docket_no, # The key for the dropdown/preview
            'csv_filename': csv_filename # Just the filename
        })

    # 6b. Export all entries to a single CSV
    combined_csv_filename = "all_dockets_report.csv" # Define a standard name
    created_combined_csv_name = export_all_entries_to_csv(all_cbb_entries, output_dir, combined_csv_filename)
    if created_combined_csv_name:
        exported_files_info.append({
            'docket_no': '_ALL_ENTRIES_', # Special identifier
            'csv_filename': created_combined_csv_name
        })
        logging.info(f"Added combined CSV '{created_combined_csv_name}' to exported files list.")
    else:
        logging.warning(f"Failed to create the combined CSV file. It will not be in the exported files list.")

    # 7. Generate images if requested
    if generate_images:
        image_output_dir = output_dir / f"{pdf_path.stem}_images"
        logging.info(f"CBB Image generation requested. Outputting images to: {image_output_dir}")
        try:
            image_output_dir.mkdir(parents=True, exist_ok=True)
            pdf_to_images(str(pdf_path), str(image_output_dir))
            logging.info(f"Successfully converted CBB PDF pages to images in {image_output_dir}")
        except ImportError as ie:
             logging.error(f"Failed to import pdf_to_images. Cannot generate images: {ie}", exc_info=True)
        except Exception as e:
            logging.error(f"Failed to convert CBB PDF to images: {e}", exc_info=True)

    logging.info(f"CBB Processing finished for {pdf_path}. Output files are in {output_dir}")

    # Return the list of dicts with docket number and CSV filename
    return exported_files_info


# --- Script Execution (if run directly) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a DHCR CBB PDF file to extract case data into CSV files per docket.")
    parser.add_argument(
        "pdf_file",
        type=str,
        help="Path to the CBB PDF file to process."
    )
    parser.add_argument(
        "-i", "--images",
        action="store_true",
        help="Generate images from the PDF pages."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save output CSVs and images (default: {DEFAULT_OUTPUT_DIR}). This directory will also contain '{combined_csv_filename}' if entries are found."
    )
    args = parser.parse_args()

    pdf_path_arg = Path(args.pdf_file)
    output_dir_arg = Path(args.output)

    if not pdf_path_arg.exists():
        logging.error(f"The specified CBB PDF file does not exist: {pdf_path_arg}")
    elif not pdf_path_arg.is_file():
         logging.error(f"The specified path is not a file: {pdf_path_arg}")
    else:
        output_dir_arg.mkdir(parents=True, exist_ok=True)
        process_cbb_pdf(pdf_path_arg, output_dir_arg, generate_images=args.images) 