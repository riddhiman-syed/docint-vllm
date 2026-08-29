"""
Retrieves relevant chunks from Qdrant for a given question.
"""
import logging
from dataclasses import dataclass

from app.config import get_settings
from app.ingestion.embedder import get_embedder
from app.ingestion.indexer import get_qdrant_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    page_number: int
    score: float


def retrieve(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    settings = get_settings()
    embedder = get_embedder()
    client = get_qdrant_client()

    query_vector = embedder.embed_query(question)
    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    chunks = [
        RetrievedChunk(
            text=point.payload["text"],
            source=point.payload["source"],
            page_number=point.payload["page_number"],
            score=point.score,
        )
        for point in results
    ]
    logger.info("Retrieved %d chunks for question: %r (top score=%.3f)", len(chunks), question[:80], chunks[0].score if chunks else 0.0)
    return chunks
