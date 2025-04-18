import os
import io
from pdf2image import convert_from_path
from PIL import Image

PDF_PATH = "dhcr_tiny.pdf" # Replace with your PDF file path
TEMP_IMAGE_DIR = "temp_images"
POPPLER_PATH = r"C:\Users\matts\Dropbox\Matt Savoca\Projects\My Projects\dhcr_parse\scratch\dependencies\poppler_x86_x64\poppler-24.08.0\Library\bin" 


def pdf_to_images(pdf_path, output_dir):
    """Converts a PDF to a series of PNG images."""
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
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