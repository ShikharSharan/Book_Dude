# splitter.py
"""
Book Dude - Text Splitter Module
--------------------------------
This module splits cleaned text into chapters and chunks.
Steps:
1. Detect chapter boundaries (regex or keywords)
2. Split into chapters
3. Further split chapters into overlapping chunks
"""

import re
from typing import List, Dict


def split_into_chapters(text: str) -> Dict[str, str]:
    """
    Splits text into chapters using regex patterns.

    Args:
        text (str): Cleaned book text.

    Returns:
        Dict[str, str]: Dictionary with chapter titles as keys and text as values.
    """
    # Regex pattern for chapter headings (e.g., "Chapter 1", "CHAPTER ONE")
    chapter_pattern = re.compile(r'(Chapter\s+\d+|CHAPTER\s+[A-Z]+)', re.IGNORECASE)

    matches = list(chapter_pattern.finditer(text))
    chapters = {}

    if not matches:
        # Fallback: treat entire text as one chapter
        chapters["Chapter 1"] = text
        return chapters

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapter_title = match.group()
        chapter_text = text[start:end].strip()
        chapters[chapter_title] = chapter_text

    return chapters


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks.

    Args:
        text (str): Input text.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of words to overlap between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def split_book(text: str, chunk_size: int = 500, overlap: int = 50) -> Dict[str, List[str]]:
    """
    Splits entire book into chapters and chunks.

    Args:
        text (str): Cleaned book text.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of words to overlap between chunks.

    Returns:
        Dict[str, List[str]]: Dictionary with chapter titles as keys and list of chunks as values.
    """
    chapters = split_into_chapters(text)
    book_chunks = {}
    for title, chapter_text in chapters.items():
        book_chunks[title] = chunk_text(chapter_text, chunk_size, overlap)
    return book_chunks


if __name__ == "__main__":
    # Example usage for local testing
    sample_text = """
    Chapter 1 Introduction
    This is some sample text for chapter one. It explains the basics.
    Chapter 2 Methods
    This chapter describes methods and approaches in detail.
    """

    chapters = split_into_chapters(sample_text)
    print("Detected Chapters:", list(chapters.keys()))

    chunks = split_book(sample_text, chunk_size=20, overlap=5)
    for title, ch_chunks in chunks.items():
        print(f"\n{title} -> {len(ch_chunks)} chunks")
        for c in ch_chunks:
            print("-", c)
