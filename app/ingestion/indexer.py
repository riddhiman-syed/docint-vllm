"""
Pushes embedded chunks into Qdrant.

Replaces the original app.py's SQLite text_chunks table (which stored
raw text only, no vectors, and required a separate similarity step)
with a proper vector index. Metadata needed for citations (source
filename, page number) is stored as Qdrant payload alongside each
vector.
"""
import logging
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import get_settings
from app.ingestion.chunker import Chunk
from app.ingestion.embedder import get_embedder

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)


def ensure_collection(client: QdrantClient, dimension: int, recreate: bool = False) -> None:
    settings = get_settings()
    name = settings.QDRANT_COLLECTION_NAME

    if recreate and client.collection_exists(name):
        logger.info("Dropping existing collection %s (recreate=True)", name)
        client.delete_collection(name)

    if not client.collection_exists(name):
        logger.info("Creating collection %s (dim=%d, cosine)", name, dimension)
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )


def index_chunks(chunks: Sequence[Chunk], recreate: bool = False, batch_size: int = 64) -> int:
    """Embed and upsert a batch of chunks into Qdrant. Returns count indexed."""
    if not chunks:
        return 0

    settings = get_settings()
    embedder = get_embedder()
    client = get_qdrant_client()
    ensure_collection(client, embedder.dimension, recreate=recreate)

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_documents([c.text for c in batch])
        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                },
            )
            for chunk, vector in zip(batch, vectors)
        ]
        client.upsert(collection_name=settings.QDRANT_COLLECTION_NAME, points=points)
        total += len(points)
        logger.info("Indexed %d/%d chunks", total, len(chunks))

    return total
