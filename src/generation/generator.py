import logging
from typing import Sequence

from openai import OpenAI

from ..config import get_settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a precise, enterprise-grade question-answering assistant.

Rules:
- Answer using only the provided context. Do not fabricate information.
- If the context is insufficient to answer, say so clearly.
- Cite the source document (filename) when referencing specific information.
- Be concise. Prefer bullet points for multi-part answers.
"""


class RAGGenerator:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def generate(self, query: str, context_chunks: Sequence[dict]) -> dict:
        if not context_chunks:
            return {
                "answer": "No relevant documents found for this query.",
                "sources": [],
                "tokens_used": 0,
            }

        context = self._format_context(context_chunks)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        sources = sorted({c["source"] for c in context_chunks})
        return {
            "answer": response.choices[0].message.content,
            "sources": sources,
            "tokens_used": response.usage.total_tokens,
        }

    @staticmethod
    def _format_context(chunks: Sequence[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{i}] Source: {chunk.get('filename', chunk.get('source', 'unknown'))}\n"
                f"{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)
