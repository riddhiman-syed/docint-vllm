"""
Generates text descriptions of extracted images via the vLLM-served
VLM, so chart/diagram/photo content becomes searchable alongside
regular text chunks.
"""
import base64
import logging
import uuid
from typing import Iterable, Iterator

from app.ingestion.chunker import Chunk
from app.ingestion.image_extractor import ExtractedImage
from app.llm.client import get_llm_client

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "Describe this image in detail, as it appears in a document. If it's a chart, graph, or "
    "table, describe the specific data, trends, axis labels, and values shown — not just that "
    "a chart is present. If it's a photo or diagram, describe what it depicts. Be factual and "
    "specific; this description will be used to answer questions about the document."
)


def caption_image(image: ExtractedImage) -> str:
    llm = get_llm_client()
    b64 = base64.b64encode(image.data).decode("utf-8")
    data_url = f"data:{image.mime_type};base64,{b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    try:
        return llm.chat(messages, temperature=0.0, max_tokens=400)
    except Exception:
        logger.exception("Failed to caption image from %s p.%d (index %d)", image.source, image.page_number, image.image_index)
        return ""


def caption_images_as_chunks(images: Iterable[ExtractedImage]) -> Iterator[Chunk]:
    """Caption each image and wrap it as a Chunk, ready for the same embed/index path as text chunks."""
    for image in images:
        caption = caption_image(image)
        if not caption.strip():
            continue
        yield Chunk(
            id=str(uuid.uuid4()),
            source=image.source,
            page_number=image.page_number,
            text=caption,
            chunk_index=image.image_index,
            chunk_type="image_caption",
        )
