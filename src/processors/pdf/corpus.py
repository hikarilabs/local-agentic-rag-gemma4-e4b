import re
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader

_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


class Corpus(BaseModel):
    full_text: str

    # list of tuples representing: (start_char_idx, end_char_idx, page_number)
    page_maps: list[tuple[int, int, int]]
    source_name: str


def load_pdf_corpus(file_path: Path) -> Corpus:
    """Loads a PDF into a continuous text stream while mapping character positions to page numbers."""
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    reader = PdfReader(file_path)
    full_text_buffer = []
    page_maps = []
    current_char_idx = 0

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        # standardise line breaks but don't strip yet
        if not text:
            continue

        start_idx = current_char_idx
        full_text_buffer.append(text)
        current_char_idx += len(text)
        end_idx = current_char_idx

        page_maps.append((start_idx, end_idx, page_number))

    if not full_text_buffer:
        raise ValueError("No text found in the PDF.")

    return Corpus(
        full_text="".join(full_text_buffer),
        page_maps=page_maps,
        source_name=file_path.name,
    )


def resolve_pages_for_chunk(
    chunk_start: int, chunk_end: int, page_maps: list[tuple[int, int, int]]
) -> list[int]:
    """Finds all page numbers that overlap with the given chunk of text."""
    pages = []
    for page_start, page_end, page_number in page_maps:
        if max(chunk_start, page_start) < min(chunk_end, page_end):
            pages.append(page_number)
    return pages
