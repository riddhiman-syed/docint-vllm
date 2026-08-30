"""
Thin wrapper around vLLM's OpenAI-compatible /v1/chat/completions endpoint.

Deliberately not using langchain's LLM wrapper here — the `openai` SDK
talking to vLLM directly is simpler, has fewer moving parts to debug,
and is exactly what vLLM's own docs recommend for compatibility.
"""
import base64
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

    def chat_with_image(self, prompt: str, image_bytes: bytes, mime_type: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        """
        Single-turn vision chat: an image plus a text prompt about it.
        Shared by app/ingestion/captioner.py's captioning path and the
        /ask-image API endpoint — same underlying vision call, different
        prompt and caller.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

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
