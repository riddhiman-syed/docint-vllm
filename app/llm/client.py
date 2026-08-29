"""
Thin wrapper around vLLM's OpenAI-compatible /v1/chat/completions endpoint.

Deliberately not using langchain's LLM wrapper here — the `openai` SDK
talking to vLLM directly is simpler, has fewer moving parts to debug,
and is exactly what vLLM's own docs recommend for compatibility.
"""
import logging
from functools import lru_cache

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_llm_client() -> "LLMClient":
    return LLMClient()


class LLMClient:
    def __init__(self):
        settings = get_settings()
        self.model = settings.VLLM_MODEL_NAME
        self._client = OpenAI(base_url=settings.VLLM_BASE_URL, api_key=settings.VLLM_API_KEY)

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """Single-turn (or pre-built multi-turn) chat completion, returns the text content."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def chat_json(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 512) -> str:
        """
        Chat completion with vLLM's guided JSON decoding, for nodes that
        need structured output (grading, routing). Falls back to a plain
        call with a "respond only with JSON" instruction if the server
        doesn't support response_format (older vLLM versions).
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or "{}"
        except Exception:
            logger.warning("response_format=json_object not supported by server, falling back to prompted JSON", exc_info=True)
            return self.chat(messages, temperature=temperature, max_tokens=max_tokens)
