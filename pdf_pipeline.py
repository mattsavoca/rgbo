import os
import requests
import json
import re
import csv
from pathlib import Path
from pydantic import BaseModel, Field, validator, ValidationError, field_validator
from typing import List, Optional, Union, Dict
from datetime import date, datetime
from google.api_core import exceptions as google_exceptions
from google import genai
from google.genai import Client, types
from collections import defaultdict
import logging
import time
from dotenv import load_dotenv # Import dotenv
import argparse # Import argparse
from pdf_to_png import pdf_to_images # Import the image conversion function

# --- Load Environment Variables --- #
load_dotenv() # Load variables from .env file

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Explicitly check if the key was loaded, primarily for early user feedback
GEMINI_API_KEY_CHECK = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY_CHECK:
    logging.info("GEMINI_API_KEY found in environment.")
else:
    logging.warning("GEMINI_API_KEY environment variable not set or not loaded from .env. Client initialization will likely fail.")
    # You might want to exit here depending on requirements
    # exit() or raise Exception("API Key not configured")

# Specify the Gemini model to use
# Use a model that supports PDF processing, like gemini-1.5-pro-latest or gemini-1.5-flash-latest
# Check the Gemini documentation for the latest recommended models supporting PDF input.
MODEL_NAME = "gemini-2.0-flash-exp" 
LEN_TIMEOUT = 1000 * 60 # (in miliseconds) 
# Directory for output CSV files when run as script
DEFAULT_OUTPUT_DIR = Path("./dhcr_output_csvs")
# DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True) # Create the directory if it doesn't exist (moved to main block)

# --- Pydantic Models ---

def parse_flexible_date(value: Optional[Union[str, date]]) -> Optional[date]:
    """Attempts to parse various date formats, returning None if parsing fails."""
    if isinstance(value, date):
        return value
    if not value or not isinstance(value, str):
        return None
    # Handle common variations or potential LLM outputs
    value = value.strip()
    if value.lower() in ["", "none", "n/a", "na"] or value.upper() == "NC":
        return None
        
    # Special handling for "%m-%d-%y" format with century inference
    m_d_y_pattern = re.compile(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2})$") # Accept hyphen or slash
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
            pass  # If this fails, continue with other formats
            
    formats_to_try = [
        "%Y-%m-%d",  # Standard ISO
        "%m/%d/%Y",  # Common US format
        "%m-%d-%Y",
        "%d-%b-%Y",  # e.g., 01-Jan-2023
        "%B %d, %Y", # e.g., January 1, 2023
        "%Y%m%d",    # Sometimes dates appear without separators
    ]
    for fmt in formats_to_try:
        try:
            # Handle potential time components if LLM includes them
            return datetime.strptime(value.split(' ')[0], fmt).date()
        except (ValueError, TypeError):
            continue
    logging.warning(f"Could not parse date: {value}")
    return None # Return None if all formats fail

def parse_currency(value: Union[str, float, int]) -> Optional[float]:
    """Removes currency symbols, commas and converts to float."""
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ["", "none", "n/a", "na"]:
            return None
        # Remove common currency symbols and commas
        cleaned_value = re.sub(r"[$,]", "", value)
        try:
            return float(cleaned_value)
        except ValueError:
            logging.warning(f"Could not parse currency value: {value}")
            return None
    return None # Return None if type is unexpected


class DHCR_Entry(BaseModel):
    apt_number: str = Field(..., alias="Apt Number", description="Unit of the Building")
    apt_status: Optional[str] = Field(None, alias="Apt Status", description="Status of the unit")
    effective_date: Optional[date] = Field(None, alias="Effective Date", description="Effective date of the lease")
    legal_reg_rent: Optional[float] = Field(None, alias="Legal Reg Rent", description="Legal Regulated Rent (USD)")
    actual_rent_paid: Optional[float] = Field(None, alias="Actual Rent Paid", description="Actual Rent Paid (USD)")
    filing_date: Optional[date] = Field(None, alias="Filing Date", description="Date in which it was filed (e.g., 'YYYY-MM-DD' or 'NC' for None)")
    tenant_name: Optional[str] = Field(None, alias="Tenant Name", description="Tenant's Legal Name")
    lease_began: Optional[date] = Field(None, alias="Lease Began", description="When lease actually began")
    lease_ends: Optional[date] = Field(None, alias="Lease Ends", description="When lease ends") # Corrected from 'Lease Began'

    # Pydantic V2 Validators
    @field_validator('effective_date', 'lease_began', 'lease_ends', 'filing_date', mode='before')
    @classmethod
    def parse_dates(cls, value):
        return parse_flexible_date(value)

    @field_validator('legal_reg_rent', 'actual_rent_paid', mode='before')
    @classmethod
    def parse_rents(cls, value):
        return parse_currency(value)

    @field_validator('apt_number', mode='before')
    @classmethod
    def normalize_apt_number(cls, value):
        if isinstance(value, str):
            return value.strip().upper() # Normalize for consistent grouping
        return value

    # Provide aliases for CSV export consistency if needed
    def to_dict_for_csv(self):
         return {
            "Apt Number": self.apt_number,
            "Apt Status": self.apt_status,
            "Effective Date": self.effective_date.isoformat() if self.effective_date else None,
            "Legal Reg Rent": self.legal_reg_rent,
            "Actual Rent Paid": self.actual_rent_paid,
            "Filing Date": self.filing_date.isoformat() if self.filing_date else None,
            "Tenant Name": self.tenant_name,
            "Lease Began": self.lease_began.isoformat() if self.lease_began else None,
            "Lease Ends": self.lease_ends.isoformat() if self.lease_ends else None,
        }

class DHCR_Page_Data(BaseModel):
    dhcr_entries: List[DHCR_Entry]


# --- Gemini Interaction Functions ---

def get_gemini_client():
    """Initializes and returns the Gemini client by explicitly passing the API key."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logging.error("Failed to get GEMINI_API_KEY from environment for client initialization.")
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        client = Client(api_key=api_key) # Pass key directly
        # Test connection by listing models (optional, uncomment to verify)
        # client.list_models()
        logging.info("Gemini client initialized successfully using provided API key.")
        return client
    except ValueError as ve:
        logging.error(f"Client Initialization Error: {ve}")
        raise # Re-raise the specific error
    except Exception as e:
        logging.error(f"Failed to initialize Gemini client: {e}")
        raise

def identify_dhcr_pages(client: Client, pdf_data: bytes) -> List[int]:
    """
    Uses Gemini to identify pages containing valid DHCR tables.

    Args:
        client: The initialized Gemini API Client.
        pdf_data: Raw bytes of the PDF file.

    Returns:
        A list of 1-based page numbers containing valid DHCR tables.
        Returns an empty list if no valid pages are found or an error occurs.
    """
    prompt = (
        "Analyze the provided PDF document, which contains rent registration information. "
        "Identify the 1-based page numbers that contain a DHCR table with actual registration data. "
        "These tables are usually centered on the page and have the following columns: 'Apartment Number', 'Apt Status', 'Effective Date', 'Legal Reg Rent', 'Actual Rent Paid', 'Filing Date', 'Tenant Name', 'Lease Began', 'Lease Ends'. "
        "Critically, **exclude** any pages where the table area primarily contains the phrase "
        "'No Information Found for this Registration Year' or similar indications of no data. "
        "Return the result ONLY as a JSON list of integer page numbers, e.g., [3, 4, 6]."
        "If no pages with valid data tables are found, return an empty list []."
    )
    max_retries = 3
    retry_delay = 5 # seconds

    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries}: Identifying DHCR pages...")
            # Call generate_content on the client.models instance
            response = client.models.generate_content( # Use client.models
                model=MODEL_NAME, # Specify model here
                contents=[
                    types.Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
                    prompt,
                ],
                config=types.GenerateContentConfig( # Use GenerateContentConfig
                    temperature=0.0, # Low temperature for factual identification
                    response_mime_type="application/json", # Request JSON output directly
                    http_options=types.HttpOptions(timeout=LEN_TIMEOUT) # Set timeout here
                ),
            )

            logging.debug(f"Raw page identification response: {response.text}")

            # Directly parse JSON response
            page_numbers = json.loads(response.text)

            if isinstance(page_numbers, list) and all(isinstance(p, int) for p in page_numbers):
                 # Ensure pages are positive and sorted
                valid_pages = sorted([p for p in page_numbers if p > 0])
                logging.info(f"Identified valid DHCR pages: {valid_pages}")
                return valid_pages
            else:
                logging.warning(f"Received unexpected format for page numbers: {page_numbers}. Trying again if possible.")
                # Continue to retry loop if format is wrong

        except json.JSONDecodeError as e:
            logging.warning(f"Attempt {attempt + 1}: Failed to decode JSON from page identification response: {e}. Response text: {response.text}")
        except (google_exceptions.DeadlineExceeded, google_exceptions.ServiceUnavailable, requests.exceptions.ReadTimeout) as e:
             logging.warning(f"Attempt {attempt + 1}: Network or timeout error during page identification: {e}. Retrying in {retry_delay}s...")
             time.sleep(retry_delay)
        except google_exceptions.ResourceExhausted as e:
             logging.error(f"Gemini API quota exceeded during page identification: {e}")
             return [] # Stop retrying on quota issues
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: An unexpected error occurred during page identification: {e}")
            # Consider retrying for some generic errors, but maybe stop for others
            # For simplicity, retry generic errors here too.
            time.sleep(retry_delay)


    logging.error("Failed to identify DHCR pages after multiple retries.")
    return []


def extract_data_from_page(client: Client, pdf_data: bytes, page_number: int) -> Optional[DHCR_Page_Data]:
    """
    Uses Gemini to extract structured DHCR data from a specific PDF page.

    Args:
        client: The initialized Gemini API Client.
        pdf_data: Raw bytes of the PDF file.
        page_number: The 1-based page number to extract data from.

    Returns:
        A DHCR_Page_Data object containing the extracted entries, or None if extraction fails.
    """
    # Generate the schema dynamically from the Pydantic model
    schema = DHCR_Entry.model_json_schema()

    prompt = (
        f"From the provided PDF document, focus **only** on page {page_number}. "
        f"Extract all rows from the DHCR table present on this page. "
        "These tables are usually centered on the page and have the following columns: 'Apartment Number', 'Apt Status', 'Effective Date', 'Legal Reg Rent', 'Actual Rent Paid', 'Filing Date', 'Tenant Name', 'Lease Began', 'Lease Ends'. "
        "Return the extracted data as a JSON object with a single key 'dhcr_entries', where the value is a list of JSON objects. "
        "Each object in the list should represent one row from the table and conform to the following JSON schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "Ensure dates are formatted as 'YYYY-MM-DD' even if the date is not provided in that format on the page (it is often provided in 'MM/DD/YY' format). If a date is missing or unclear, represent it as null. "
        "Take extra consideration aligning the correct values/dates to the correct variable names, especially in cases where there are blanks or missing values proceeding or preceeding it in the row."
        "If 'Filing Date' is 'NC', represent it as null. "
        "Normalize apartment numbers (e.g., '1A' instead of ' 1a '). "
        "Parse monetary values (Legal Reg Rent, Actual Rent Paid) as numbers, removing any '$' or ',' symbols. Represent missing monetary values as null. "
        "If the table is empty or cannot be parsed on this specific page, return {\"dhcr_entries\": []}."
    )

    max_retries = 3
    retry_delay = 5 # seconds

    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries}: Extracting data from page {page_number}...")
            # Call generate_content on the client.models instance
            response = client.models.generate_content( # Use client.models
                model=MODEL_NAME, # Specify model here
                contents=[
                    # Send only the relevant page if API supports it, otherwise send whole PDF
                    # Current Gemini API (as of early 2024) generally takes the whole PDF
                    # and relies on the prompt to specify the page.
                    types.Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
                    prompt,
                ],
                 config=types.GenerateContentConfig( # Use GenerateContentConfig
                    temperature=0.0, # Low temperature for extraction accuracy
                    response_mime_type="application/json", # Request JSON output
                    http_options=types.HttpOptions(timeout=LEN_TIMEOUT) # Set timeout here
                ),
            )

            logging.debug(f"Raw data extraction response for page {page_number}: {response.text}")

            # Try to load the JSON response directly
            raw_data = json.loads(response.text)

            # Validate the structure before parsing with Pydantic
            if isinstance(raw_data, dict) and "dhcr_entries" in raw_data and isinstance(raw_data["dhcr_entries"], list):
                 # Use Pydantic to parse and validate the list of entries
                 page_data = DHCR_Page_Data(dhcr_entries=raw_data["dhcr_entries"])
                 logging.info(f"Successfully extracted and validated {len(page_data.dhcr_entries)} entries from page {page_number}.")
                 return page_data
            else:
                 logging.warning(f"Attempt {attempt + 1}: Unexpected JSON structure received for page {page_number}: {raw_data}. Retrying...")
                 # Continue retry loop

        except json.JSONDecodeError as e:
            logging.warning(f"Attempt {attempt + 1}: Failed to decode JSON from page {page_number} response: {e}. Response text: {response.text}")
        except ValidationError as e:
            logging.warning(f"Attempt {attempt + 1}: Pydantic validation failed for page {page_number}: {e}. Data: {response.text}")
        except (google_exceptions.DeadlineExceeded, google_exceptions.ServiceUnavailable, requests.exceptions.ReadTimeout) as e:
             logging.warning(f"Attempt {attempt + 1}: Network or timeout error during extraction for page {page_number}: {e}. Retrying in {retry_delay}s...")
             time.sleep(retry_delay)
        except google_exceptions.ResourceExhausted as e:
             logging.error(f"Gemini API quota exceeded during extraction for page {page_number}: {e}")
             return None # Stop retrying on quota issues
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: An unexpected error occurred during extraction for page {page_number}: {e}")
            # Consider retrying for some generic errors
            time.sleep(retry_delay)

    logging.error(f"Failed to extract data from page {page_number} after multiple retries.")
    return None

# --- Data Processing and Export ---

def export_unit_to_csv(apt_number: str, unit_data: List[DHCR_Entry], output_dir: Path):
    """Exports the data for a single apartment unit to a CSV file in the specified directory."""
    if not unit_data:
        logging.warning(f"No data found for Apt Number: {apt_number}. Skipping CSV export.")
        return

    # Sort data by effective date (earliest to latest)
    # Handle None dates by placing them at the beginning or end (here, beginning)
    unit_data.sort(key=lambda x: x.effective_date or date.min)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure the target directory exists

    # Sanitize apartment number for filename
    safe_apt_number = re.sub(r'[\\/*?:"<>|]', "_", apt_number) # Replace invalid filename characters
    filename = output_dir / f"apt_{safe_apt_number}.csv" # Use the passed output_dir

    logging.info(f"Exporting data for Apt {apt_number} to {filename}")

    # Use the first entry to get headers in the desired order
    # Or define headers explicitly if order matters strongly
    # headers = list(DHCR_Entry.model_fields.keys()) # Gets internal field names
    headers = list(unit_data[0].to_dict_for_csv().keys()) # Use aliased names

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for entry in unit_data:
                 # Convert entry to dict using aliases for consistency
                 writer.writerow(entry.to_dict_for_csv())
    except IOError as e:
        logging.error(f"Failed to write CSV file for Apt {apt_number}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during CSV export for Apt {apt_number}: {e}")


# --- Main Execution ---

def process_pdf(pdf_filepath: Union[str, Path], output_dir: Path, generate_images: bool = False):
    """Main function to process a single PDF, outputting results to the specified directory."""
    pdf_path = Path(pdf_filepath)
    if not pdf_path.is_file():
        logging.error(f"PDF file not found: {pdf_path}")
        return

    logging.info(f"Processing PDF: {pdf_path}")

    # 1. Load PDF data
    try:
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        logging.info(f"Loaded PDF data ({len(pdf_data)} bytes).")
    except IOError as e:
        logging.error(f"Failed to read PDF file {pdf_path}: {e}")
        return

    # 2. Initialize Gemini Client
    try:
        client = get_gemini_client()
    except Exception:
        # Error already logged in get_gemini_client
        return

    # 3. Identify relevant pages
    valid_pages = identify_dhcr_pages(client, pdf_data)
    if not valid_pages:
        logging.warning(f"No pages with valid DHCR data found in {pdf_path}. Exiting.")
        return

    # 4. Extract data from each relevant page
    building_dhcr_db: List[DHCR_Entry] = []
    for page_num in valid_pages:
        page_data = extract_data_from_page(client, pdf_data, page_num)
        if page_data and page_data.dhcr_entries:
            building_dhcr_db.extend(page_data.dhcr_entries)
            logging.info(f"Added {len(page_data.dhcr_entries)} entries from page {page_num} to the database.")
        else:
            logging.warning(f"Could not extract valid data from page {page_num}.")

    if not building_dhcr_db:
        logging.warning("No DHCR entries were successfully extracted from any identified page.")
        return # Return early if no data extracted

    logging.info(f"Total extracted DHCR entries across all pages: {len(building_dhcr_db)}")

    # 5. Group data by Apartment Number
    grouped_by_apt: Dict[str, List[DHCR_Entry]] = defaultdict(list)
    for entry in building_dhcr_db:
        # Ensure apt_number is valid before grouping
        if entry.apt_number and isinstance(entry.apt_number, str):
             # Apt number already normalized by Pydantic validator
             grouped_by_apt[entry.apt_number].append(entry)
        else:
             logging.warning(f"Skipping entry with invalid or missing apt_number: {entry}")


    # 6. Export each unit's data to CSV, sorted by date
    logging.info(f"Found {len(grouped_by_apt)} unique apartment units. Exporting to CSV in {output_dir}...")
    for apt_number, unit_entries in grouped_by_apt.items():
        export_unit_to_csv(apt_number, unit_entries, output_dir) # Pass output_dir here

    # 7. Generate images if requested
    if generate_images:
        image_output_dir = output_dir / f"{pdf_path.stem}_images"
        logging.info(f"Image generation requested. Outputting images to: {image_output_dir}")
        try:
            # pdf_to_images needs the output dir created
            image_output_dir.mkdir(parents=True, exist_ok=True)
            # Ensure the function receives Path objects or strings as expected
            pdf_to_images(str(pdf_path), str(image_output_dir))
            logging.info(f"Successfully converted PDF pages to images in {image_output_dir}")
        except ImportError as ie:
             logging.error(f"Failed to convert PDF to images. pdf2image or its dependency (Poppler) might be missing: {ie}")
             logging.error("Please ensure pdf2image is installed (`pip install pdf2image`) and Poppler is installed and in your system's PATH.")
        except Exception as e:
            logging.error(f"Failed to convert PDF to images: {e}")
            # Continue even if image generation fails


    logging.info(f"Processing finished for {pdf_path}. Output files are in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a DHCR PDF file to extract rent registration data into CSV files per unit.")
    parser.add_argument(
        "pdf_file",
        type=str,
        help="Path to the DHCR PDF file to process."
    )
    parser.add_argument(
        "-i", "--images",
        action="store_true",
        help="Generate images from the PDF."
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_file)

    if not pdf_path.exists():
        logging.error(f"The specified PDF file does not exist: {pdf_path}")
    elif not pdf_path.is_file():
         logging.error(f"The specified path is not a file: {pdf_path}")
    else:
        # Create the default output directory when run as script
        DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)

        # Pass the default output dir and image flag to process_pdf
        process_pdf(pdf_path, DEFAULT_OUTPUT_DIR, generate_images=args.images)