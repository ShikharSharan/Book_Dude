# cleaner.py
"""
Book Dude - Text Cleaner Module
-------------------------------
This module cleans and normalizes raw text extracted from PDF files.
It provides functions to:
- Remove headers, footers, and page numbers
- Normalize whitespace and punctuation
- Strip unwanted characters
"""

import re

def clean_text(raw_text: str) -> str:
    """
    Cleans raw text extracted from a PDF.

    Args:
        raw_text (str): The raw text extracted from the PDF.

    Returns:
        str: Cleaned and normalized text.
    """
    if not raw_text:
        return ""

    # Remove page markers like "--- Page X ---"
    text = re.sub(r'--- Page \d+ ---', ' ', raw_text)

    # Remove page numbers (standalone digits)
    text = re.sub(r'\b\d+\b', ' ', text)

    # Remove multiple spaces, tabs, and newlines
    text = re.sub(r'\s+', ' ', text)

    # Normalize punctuation spacing
    text = re.sub(r'\s([?.!,"])', r'\1', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def clean_chapter(chapter_text: str) -> str:
    """
    Cleans text for a single chapter or section.

    Args:
        chapter_text (str): Raw text of a chapter.

    Returns:
        str: Cleaned chapter text.
    """
    return clean_text(chapter_text)


if __name__ == "__main__":
    # Example usage for local testing
    sample_raw = """
    --- Page 1 ---
    Chapter 1: Introduction
    This is   a sample   text with   irregular spacing.
    Page 1
    --- Page 2 ---
    More text here... Page 2
    """
    cleaned = clean_text(sample_raw)
    print("Cleaned Text:\n", cleaned)
