from dataclasses import dataclass, field
from typing import Iterator

from ..config import get_settings

# Ordered from coarsest to finest — try each separator before splitting mid-sentence.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    source: str
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}_{self.chunk_index}"


def chunk_document(doc: dict) -> Iterator[Chunk]:
    """Split a document dict into overlapping Chunk objects."""
    settings = get_settings()
    texts = _split(doc["content"], settings.chunk_size, settings.chunk_overlap)
    for i, text in enumerate(texts):
        if len(text) < settings.min_chunk_length:
            continue
        yield Chunk(
            text=text,
            doc_id=doc["doc_id"],
            chunk_index=i,
            source=doc["source"],
            metadata={"filename": doc["filename"]},
        )


def _split(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a natural boundary rather than mid-word.
        if end < len(text):
            for sep in _SEPARATORS:
                pos = text.rfind(sep, start, end)
                if pos > start:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks
