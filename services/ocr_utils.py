import pytesseract
from PIL import Image

from pdf2image import convert_from_path  # or pdf2image to handle PDF -> image

from config import TESSERACT_CMD

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def pdf_to_images(pdf_path, dpi=300):
    """
    Convert each page of a PDF to a list of PIL images for OCR.
    """
    pages = convert_from_path(pdf_path, dpi)
    return pages

def ocr_image(image):
    """
    Perform OCR on a single PIL image.
    """
    text = pytesseract.image_to_string(image)
    return text