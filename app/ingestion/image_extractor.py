"""
Extracts embedded raster images from PDF pages, via PyMuPDF, for
multimodal (image-aware) ingestion.

Scope/limitation worth knowing: this captures embedded raster images
(logos, photos, scanned charts saved as images). It does NOT capture
vector-drawn graphics — charts/diagrams drawn directly with PDF drawing
operators rather than embedded as an image file. PyMuPDF can render
full pages to catch those too (page.get_pixmap()), at a higher
captioning cost per page — a reasonable next step if this proves too
narrow for a given document set.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Union

import pymupdf

logger = logging.getLogger(__name__)

_EXT_TO_MIME = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
    "bmp": "image/bmp", "tiff": "image/tiff", "tif": "image/tiff", "gif": "image/gif",
}


@dataclass(frozen=True)
class ExtractedImage:
    source: str
    page_number: int
    image_index: int
    data: bytes
    mime_type: str


def _open(pdf_source: Union[str, Path, BinaryIO]) -> pymupdf.Document:
    if isinstance(pdf_source, (str, Path)):
        return pymupdf.open(pdf_source)
    return pymupdf.open(stream=pdf_source.read(), filetype="pdf")


def extract_images(
    pdf_source: Union[str, Path, BinaryIO],
    source_name: str | None = None,
    min_size_bytes: int = 2048,
) -> Iterator[ExtractedImage]:
    """
    min_size_bytes filters out tiny embedded images (icons, bullets,
    logos) that aren't worth captioning and would just add noise.
    """
    if source_name is None:
        if isinstance(pdf_source, (str, Path)):
            source_name = Path(pdf_source).name
        else:
            raise ValueError("source_name is required when pdf_source is a file-like object")

    doc = _open(pdf_source)
    try:
        for page_num, page in enumerate(doc, start=1):
            for idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    logger.exception("Failed to extract image xref=%s on %s p.%d", xref, source_name, page_num)
                    continue

                data = base_image["image"]
                if len(data) < min_size_bytes:
                    logger.debug("Skipping tiny image on %s p.%d (%d bytes)", source_name, page_num, len(data))
                    continue

                ext = base_image.get("ext", "png").lower()
                mime_type = _EXT_TO_MIME.get(ext, "image/png")
                yield ExtractedImage(source=source_name, page_number=page_num, image_index=idx, data=data, mime_type=mime_type)
    finally:
        doc.close()


def extract_images_from_directory(directory: Union[str, Path], glob_pattern: str = "**/*.pdf") -> Iterator[ExtractedImage]:
    directory = Path(directory)
    for pdf_path in sorted(directory.glob(glob_pattern)):
        try:
            yield from extract_images(pdf_path)
        except Exception:
            logger.exception("Failed to extract images from %s — skipping", pdf_path)
