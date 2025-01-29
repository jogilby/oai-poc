import os
import PyPDF2
import pdfplumber
from .ocr_utils import pdf_to_images, ocr_image


def extract_text_from_pdf(pdf_path):
    """
    Attempt to extract text directly. If no text is found, fallback to OCR.
    Returns the full text of the PDF.
    """
    # Attempt extraction with PyPDF2
    pdf_reader = PyPDF2.PdfReader(open(pdf_path, "rb"))
    extracted_text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            extracted_text += page.extract_text()
    
    # If PyPDF2 text is too short, try with pdfplumber (more reliable in some cases)
    if len(extracted_text.strip()) < 50:  # heuristic
        with pdfplumber.open(pdf_path) as pdf:
            extracted_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text

    # If still no text, fallback to OCR
    if len(extracted_text.strip()) < 50:
        extracted_text = ""
        images = pdf_to_images(pdf_path)
        for img in images:
            extracted_text += ocr_image(img)
    
    return extracted_text


def chunk_text(text, max_tokens=500):
    """
    Split text into chunks of size `max_tokens`.
    For simplicity, we will approximate tokens by words or characters.
    You could do a more advanced approach with GPT token counting.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_count = 0

    for word in words:
        current_chunk.append(word)
        current_count += 1
        if current_count >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_count = 0
    
    # Add the last chunk if any words remain
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks