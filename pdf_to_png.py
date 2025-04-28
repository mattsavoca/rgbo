import os
import io
import logging # Import logging
# Remove pdf2image imports
# from pdf2image import convert_from_path, pdfinfo_from_path
# from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
import fitz # Import PyMuPDF
#from PIL import Image

PDF_PATH = "dhcr_tiny.pdf" # Replace with your PDF file path
TEMP_IMAGE_DIR = "temp_images"
# Remove Poppler path related variables
# POPPLER_ENV_VAR = "POPPLER_BIN_PATH" 

# Update function signature: remove poppler_path argument
def pdf_to_images(pdf_path, output_dir):
    """Converts a PDF to a series of PNG images using PyMuPDF (fitz)."""
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []

    # Remove Poppler path determination logic
    # ... existing code ...

    try:
        # Open the PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        logging.info(f"Opened PDF {pdf_path} with PyMuPDF. Pages: {len(doc)}")

        # Check if PDF is empty
        if len(doc) == 0:
            logging.warning(f"PDF file {pdf_path} has 0 pages. No images will be generated.")
            doc.close()
            return []

        # Iterate through pages and save as PNG
        for i, page in enumerate(doc):
            try:
                # Render page to a pixmap (higher DPI for better quality)
                pix = page.get_pixmap(dpi=200) 
                image_path = os.path.join(output_dir, f"page_{i+1}.png")
                pix.save(image_path, "png")
                image_paths.append(image_path)
                logging.debug(f"Saved page {i+1} to {image_path}")
            except Exception as page_err:
                 logging.error(f"Error processing page {i+1} of {pdf_path}: {page_err}", exc_info=True)
                 # Decide whether to continue or raise; continuing for now

        doc.close() # Close the document
        logging.info(f"Successfully converted PDF {pdf_path} to {len(image_paths)} images in {output_dir}")

    except fitz.fitz.FileNotFoundError: # PyMuPDF specific exception
        logging.error(f"PDF file not found at path: {pdf_path}")
        raise # Reraise for the caller
    except fitz.fitz.FZ_ERROR_GENERIC as fitz_err: # Catch generic MuPDF errors
         logging.error(f"PyMuPDF error processing file {pdf_path}: {fitz_err}", exc_info=True)
         # Check if it's a known issue like needing password
         if "needs a password" in str(fitz_err).lower():
             logging.error(f"The PDF file {pdf_path} is password protected.")
         # Raise a generic error for the caller
         raise RuntimeError(f"Failed to process PDF {pdf_path} with PyMuPDF.") from fitz_err
    except Exception as e:
        # Catch other potential errors during conversion
        logging.error(f"Unexpected error during PDF conversion process with PyMuPDF: {e}", exc_info=True)
        raise # Reraise the exception to be caught by the caller
    
    return image_paths

# if __name__ == "__main__":
#     input_pdf_path = PDF_PATH
#     # Create output directory name based on input PDF name
#     base_name = os.path.basename(input_pdf_path)
#     file_name_without_ext, _ = os.path.splitext(base_name)
#     output_directory = f"{file_name_without_ext}_imgs"
# 
#     print(f"Converting {input_pdf_path} to images in {output_directory}...")
#     # Call the updated function without poppler_path
#     image_paths = pdf_to_images(input_pdf_path, output_directory) 
#     print(f"Successfully converted PDF to {len(image_paths)} images:")
#     for path in image_paths:
#         print(f"  - {path}")