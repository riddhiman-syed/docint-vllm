"""
PDF text extraction.

Kept deliberately simple and dependency-light (pypdf only). Each page's
text is yielded separately with its 1-indexed page number so downstream
chunking can attach accurate page citations — this is the piece of the
original app.py worth keeping, just extracted into a standalone,
testable function instead of being inlined in the Gradio callback.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Union

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageText:
    source: str          # filename or identifier for the originating PDF
    page_number: int      # 1-indexed
    text: str


def extract_pages(pdf_source: Union[str, Path, BinaryIO], source_name: str | None = None) -> Iterator[PageText]:
    """
    Extract text page-by-page from a single PDF.

    pdf_source: a file path, or a file-like object (e.g. an UploadFile's
                .file, or a BytesIO) — pypdf accepts both.
    source_name: identifier to tag chunks with. Defaults to the path's
                 filename if pdf_source is a path; required otherwise.
    """
    if source_name is None:
        if isinstance(pdf_source, (str, Path)):
            source_name = Path(pdf_source).name
        else:
            raise ValueError("source_name is required when pdf_source is a file-like object")

    reader = PdfReader(pdf_source)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            logger.debug("Page %d of %s had no extractable text (likely scanned/image-only)", i + 1, source_name)
            continue
        yield PageText(source=source_name, page_number=i + 1, text=text)


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
