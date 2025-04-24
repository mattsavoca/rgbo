import os
import io
import logging # Import logging
from pdf2image import convert_from_path, pdfinfo_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
from PIL import Image

PDF_PATH = "dhcr_tiny.pdf" # Replace with your PDF file path
TEMP_IMAGE_DIR = "temp_images"
# Remove the hardcoded POPPLER_PATH constant, as it's now handled via env var/argument
# POPPLER_PATH = r".\\poppler_x86_x64\\poppler-24.08.0\\Library\\bin" 
POPPLER_ENV_VAR = "POPPLER_BIN_PATH" # The environment variable app.py sets


def pdf_to_images(pdf_path, output_dir, poppler_path=None):
    """Converts a PDF to a series of PNG images."""
    os.makedirs(output_dir, exist_ok=True)

    # Determine the Poppler path to use
    path_to_use = poppler_path # Prioritize the argument
    if not path_to_use:
        path_to_use = os.environ.get(POPPLER_ENV_VAR)
        if path_to_use:
            logging.info(f"Using Poppler path from environment variable {POPPLER_ENV_VAR}: {path_to_use}")
        else:
            logging.warning(f"Poppler path argument not provided and environment variable {POPPLER_ENV_VAR} not set. Relying on system PATH.")
            # path_to_use remains None, pdf2image will try system PATH

    try:
        # First, check if Poppler is accessible and PDF is valid using pdfinfo
        try:
             pdfinfo = pdfinfo_from_path(pdf_path, poppler_path=path_to_use)
             logging.info(f"PDF Info: {pdfinfo}") # Log info like page count
        except (PDFInfoNotInstalledError, FileNotFoundError) as e:
             logging.error(f"Poppler not found or not installed correctly. Checked path: {path_to_use}. Error: {e}")
             logging.error("Ensure Poppler binaries are in the specified path or system PATH.")
             raise RuntimeError("Poppler not found, cannot generate images.") from e
        except PDFPageCountError as e:
            logging.error(f"Could not determine page count for PDF: {pdf_path}. Error: {e}")
            raise RuntimeError("Invalid PDF or could not read page count.") from e
        except PDFSyntaxError as e:
             logging.error(f"PDF syntax error in file: {pdf_path}. Error: {e}")
             raise RuntimeError("PDF file appears corrupted.") from e

        logging.info(f"Converting PDF {pdf_path} using Poppler path: {path_to_use}")
        images = convert_from_path(pdf_path, poppler_path=path_to_use)

    except Exception as e:
        # Catch other potential errors during conversion
        logging.error(f"Error during PDF conversion process: {e}", exc_info=True)
        raise # Reraise the exception to be caught by the caller
    
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i+1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    
    return image_paths

# if __name__ == "__main__":
#     input_pdf_path = PDF_PATH
#     # Create output directory name based on input PDF name
#     base_name = os.path.basename(input_pdf_path)
#     file_name_without_ext, _ = os.path.splitext(base_name)
#     output_directory = f"{file_name_without_ext}_imgs"

#     print(f"Converting {input_pdf_path} to images in {output_directory}...")
#     image_paths = pdf_to_images(input_pdf_path, output_directory)
#     print(f"Successfully converted PDF to {len(image_paths)} images:")
#     for path in image_paths:
#         print(f"  - {path}")