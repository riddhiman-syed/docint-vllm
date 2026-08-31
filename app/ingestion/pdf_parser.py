"""
PDF text extraction, via PyMuPDF.

Swapped from pypdf for extraction speed and quality on complex layouts.
Each page's text is yielded separately with its 1-indexed page number
so downstream chunking can attach accurate page citations.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Union

import pymupdf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageText:
    source: str          # filename or identifier for the originating PDF
    page_number: int      # 1-indexed
    text: str


def _open(pdf_source: Union[str, Path, BinaryIO]) -> pymupdf.Document:
    if isinstance(pdf_source, (str, Path)):
        return pymupdf.open(pdf_source)
    return pymupdf.open(stream=pdf_source.read(), filetype="pdf")


def extract_pages(pdf_source: Union[str, Path, BinaryIO], source_name: str | None = None) -> Iterator[PageText]:
    """
    Extract text page-by-page from a single PDF.

    pdf_source: a file path, or a file-like object (e.g. an UploadFile's
                .file, or a BytesIO) — PyMuPDF accepts both.
    source_name: identifier to tag chunks with. Defaults to the path's
                 filename if pdf_source is a path; required otherwise.
    """
    if source_name is None:
        if isinstance(pdf_source, (str, Path)):
            source_name = Path(pdf_source).name
        else:
            raise ValueError("source_name is required when pdf_source is a file-like object")

    doc = _open(pdf_source)
    try:
        for i, page in enumerate(doc):
            text = page.get_text()
            if not text or not text.strip():
                logger.debug("Page %d of %s had no extractable text (likely scanned/image-only)", i + 1, source_name)
                continue
            yield PageText(source=source_name, page_number=i + 1, text=text)
    finally:
        doc.close()


def extract_pages_from_directory(directory: Union[str, Path], glob_pattern: str = "**/*.pdf") -> Iterator[PageText]:
    """Walk a directory and extract pages from every PDF found in it."""
    directory = Path(directory)
    pdf_paths = sorted(directory.glob(glob_pattern))
    if not pdf_paths:
        logger.warning("No PDFs found under %s matching %s", directory, glob_pattern)
    for pdf_path in pdf_paths:
        try:
            yield from extract_pages(pdf_path)
        except Exception:
            logger.exception("Failed to parse %s — skipping", pdf_path)
