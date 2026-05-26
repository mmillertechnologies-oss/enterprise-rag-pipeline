import logging
import time
from typing import Sequence

from openai import OpenAI, RateLimitError

from ..config import get_settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100
_MAX_RETRIES = 3
_RETRY_DELAY = 5.0


class Embedder:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches with retry on rate limit."""
        results: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = list(texts[i : i + _BATCH_SIZE])
            results.extend(self._embed_with_retry(batch))
            if i + _BATCH_SIZE < len(texts):
                time.sleep(0.1)  # stay under TPM limits
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._embed_with_retry([text])[0]

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.embeddings.create(
                    model=self._model, input=texts
                )
                return [item.embedding for item in response.data]
            except RateLimitError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = _RETRY_DELAY * (2**attempt)
                logger.warning("Rate limited — retrying in %.0fs", wait)
                time.sleep(wait)
        raise RuntimeError("Embedding failed after retries")
