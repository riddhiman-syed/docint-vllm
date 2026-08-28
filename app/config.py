"""
Application settings, loaded entirely from environment variables / .env.
No secrets or connection strings are ever hardcoded here — see .env.example
for the full list of variables this app expects.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- vLLM (self-hosted, OpenAI-compatible) ---
    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    VLLM_API_KEY: str = "not-needed"  # vLLM ignores this unless --api-key is set server-side

    # --- Embeddings ---
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DEVICE: str = "cuda"  # falls back to "cpu" automatically if no GPU is visible
    EMBEDDING_BATCH_SIZE: int = 32

    # --- Chunking ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # --- Qdrant ---
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "document_chunks"

    # --- App ---
    LOG_LEVEL: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Cached so we don't re-parse the environment on every call."""
    return Settings()
