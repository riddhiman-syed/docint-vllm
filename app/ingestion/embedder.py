"""
Open-source embeddings via sentence-transformers, replacing the
original app.py's langchain_openai.OpenAIEmbeddings (paid API — not
allowed per the assignment).

BAAI/bge-base-en-v1.5 is the default: strong retrieval quality for its
size, 768-dim, and small enough to run comfortably alongside the vLLM
process on a single GPU. Swap EMBEDDING_MODEL_NAME in .env for a
different model (e.g. a "-small" variant) if VRAM is tight.
"""
import logging
from functools import lru_cache
from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

# bge-* models are trained to expect this instruction prefix on queries
# (not on the documents being indexed) — omitting it measurably hurts
# retrieval quality for this model family.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache
def get_embedder() -> "Embedder":
    return Embedder()


class Embedder:
    def __init__(self):
        settings = get_settings()
        device = settings.EMBEDDING_DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("EMBEDDING_DEVICE=cuda but no GPU is visible — falling back to cpu")
            device = "cpu"

        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.model = SentenceTransformer(self.model_name, device=device)
        # get_embedding_dimension() is the current name; fall back for
        # older sentence-transformers versions that only have the old one.
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Loaded embedding model %s (dim=%d) on %s", self.model_name, self.dimension, device)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunk text for indexing — no instruction prefix."""
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,  # required for cosine similarity search in Qdrant
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question for retrieval — uses the bge query prefix."""
        prefixed = f"{BGE_QUERY_INSTRUCTION}{text}" if "bge" in self.model_name.lower() else text
        embedding = self.model.encode(prefixed, normalize_embeddings=True)
        return embedding.tolist()
