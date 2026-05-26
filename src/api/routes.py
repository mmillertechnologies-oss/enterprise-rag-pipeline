import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from ..config import get_settings
from ..generation.generator import RAGGenerator
from ..ingestion.chunker import chunk_document
from ..ingestion.loader import load_directory, load_file
from ..retrieval.retriever import VectorRetriever
from .schemas import IngestResponse, QueryRequest, QueryResponse, StatusResponse

logger = logging.getLogger(__name__)

_retriever: VectorRetriever | None = None
_generator: RAGGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _generator
    _retriever = VectorRetriever()
    _generator = RAGGenerator()
    logger.info("RAG pipeline ready — %d chunks indexed", _retriever.document_count)
    yield


app = FastAPI(
    title="Enterprise RAG API",
    description="Production RAG pipeline for enterprise document Q&A.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    chunks = _retriever.retrieve(request.query, k=request.k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found.")
    result = _generator.generate(request.query, chunks)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        chunks_retrieved=len(chunks),
        tokens_used=result["tokens_used"],
    )


@app.post("/ingest/directory", response_model=IngestResponse)
async def ingest_directory(path: str = Query(...)) -> IngestResponse:
    chunks_total = 0
    docs_processed = 0
    for doc in load_directory(path):
        chunks = list(chunk_document(doc))
        _retriever.index(chunks)
        chunks_total += len(chunks)
        docs_processed += 1
    return IngestResponse(chunks_indexed=chunks_total, documents_processed=docs_processed)


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(path: str = Query(...)) -> IngestResponse:
    doc = load_file(path)
    chunks = list(chunk_document(doc))
    _retriever.index(chunks)
    return IngestResponse(chunks_indexed=len(chunks), documents_processed=1)


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    settings = get_settings()
    return StatusResponse(
        status="ok",
        documents_indexed=_retriever.document_count,
        model=settings.openai_model,
        embedding_model=settings.embedding_model,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
