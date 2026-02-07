# parser.py
"""
Book Dude - PDF Parser Module
-----------------------------
This module handles extraction of text from PDF files using PyPDF2.
It provides functions to:
- Read PDF files
- Extract text page by page
- Return the full text as a string
"""

from PyPDF2 import PdfReader
import re

def extract_text(pdf_file):
    """
    Extracts text from a PDF file.

    Args:
        pdf_file (str or file-like object): Path to the PDF file or uploaded file object.

    Returns:
        str: Extracted text from the entire PDF.
    """
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Clean up whitespace and add page breaks
                page_text = re.sub(r'\s+', ' ', page_text)
                text += f"\n--- Page {page_num+1} ---\n{page_text}\n"
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Error parsing PDF: {e}")

def extract_pages(pdf_file):
    """
    Extracts text from each page separately.

    Args:
        pdf_file (str or file-like object): Path to the PDF file or uploaded file object.

    Returns:
        dict: Dictionary with page numbers as keys and text as values.
    """
    try:
        reader = PdfReader(pdf_file)
        pages = {}
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                page_text = re.sub(r'\s+', ' ', page_text)
                pages[page_num+1] = page_text
        return pages
    except Exception as e:
        raise RuntimeError(f"Error parsing PDF pages: {e}")

if __name__ == "__main__":
    # Example usage for local testing
    sample_pdf = "sample_data/example.pdf"
    text = extract_text(sample_pdf)
    print("Extracted Text:\n", text[:1000])  # Print first 1000 characters
