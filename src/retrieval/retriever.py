import logging
from typing import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import get_settings
from ..embeddings.embedder import Embedder
from ..ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(self) -> None:
        settings = get_settings()
        self._k = settings.retrieval_k
        self._threshold = settings.retrieval_score_threshold
        self._embedder = Embedder()

        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, chunks: Sequence[Chunk]) -> int:
        """Embed and upsert chunks. Returns number of chunks indexed."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_batch(texts)

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": c.source, **c.metadata} for c in chunks],
        )
        logger.info("Indexed %d chunks", len(chunks))
        return len(chunks)

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """Return top-k chunks above the score threshold for query."""
        query_embedding = self._embedder.embed_query(query)
        n = k or self._k

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist  # cosine distance → similarity
            if score >= self._threshold:
                hits.append(
                    {
                        "text": doc,
                        "source": meta.get("source", ""),
                        "filename": meta.get("filename", ""),
                        "score": round(score, 4),
                    }
                )

        return hits

    @property
    def document_count(self) -> int:
        return self._collection.count()
