#!/usr/bin/env python
"""
Ingestion CLI: parse PDFs under a directory, chunk them, embed them,
and index them into Qdrant.

Usage:
    python scripts/ingest.py --source pdf/Medical --recreate
    python scripts/ingest.py --source pdf/ --glob "**/*.pdf"
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.ingestion.captioner import caption_images_as_chunks
from app.ingestion.chunker import chunk_pages
from app.ingestion.image_extractor import extract_images_from_directory
from app.ingestion.indexer import index_chunks
from app.ingestion.pdf_parser import extract_pages_from_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Directory to scan for PDFs")
    parser.add_argument("--glob", default="**/*.pdf", help="Glob pattern for PDFs within --source")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the Qdrant collection first")
    parser.add_argument("--skip-images", action="store_true", help="Skip image captioning (text chunks only, faster)")
    args = parser.parse_args()

    logging.basicConfig(level=get_settings().LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("ingest")

    if not args.source.exists():
        logger.error("Source directory does not exist: %s", args.source)
        raise SystemExit(1)

    t0 = time.perf_counter()
    pages = extract_pages_from_directory(args.source, glob_pattern=args.glob)
    chunks = list(chunk_pages(pages))
    logger.info("Parsed and chunked into %d text chunks from %s", len(chunks), args.source)

    if not args.skip_images:
        images = list(extract_images_from_directory(args.source, glob_pattern=args.glob))
        logger.info("Found %d embedded images — captioning (this calls the VLM once per image)...", len(images))
        image_chunks = list(caption_images_as_chunks(images))
        logger.info("Captioned %d/%d images into indexable chunks", len(image_chunks), len(images))
        chunks.extend(image_chunks)

    if not chunks:
        logger.warning("No chunks produced — nothing to index")
        return

    indexed = index_chunks(chunks, recreate=args.recreate)
    elapsed = time.perf_counter() - t0
    logger.info("Indexed %d chunks in %.1fs (%.1f chunks/sec)", indexed, elapsed, indexed / elapsed if elapsed else 0)


if __name__ == "__main__":
    main()
