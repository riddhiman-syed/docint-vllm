"""
Splits extracted page text into overlapping chunks for embedding.

Deliberate change from the original app.py: chunks are keyed by a
generated UUID, not by their own text content. The original used the
chunk's cleaned text as a dict key, which silently merged any two
chunks that happened to have identical text (even from different PDFs)
into a single entry with a combined page-number list. That's a
correctness bug waiting to surface on template-heavy documents
(headers, boilerplate clauses, etc.) — here, every chunk is scoped to
exactly the one page it came from, and identical text across pages is
represented as separate chunks/points.
"""
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.ingestion.pdf_parser import PageText


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page_number: int
    text: str
    chunk_index: int  # position of this chunk within its source page, for debugging/ordering


def chunk_pages(pages: Iterable[PageText], chunk_size: int | None = None, chunk_overlap: int | None = None) -> Iterator[Chunk]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        length_function=len,
    )

    for page in pages:
        cleaned = page.text.replace("\n", " ").strip()
        if not cleaned:
            continue
        for idx, piece in enumerate(splitter.split_text(cleaned)):
            yield Chunk(
                id=str(uuid.uuid4()),
                source=page.source,
                page_number=page.page_number,
                text=piece,
                chunk_index=idx,
            )
