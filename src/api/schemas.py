from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int | None = Field(None, ge=1, le=20)


class SourceChunk(BaseModel):
    text: str
    source: str
    filename: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_retrieved: int
    tokens_used: int


class IngestResponse(BaseModel):
    chunks_indexed: int
    documents_processed: int


class StatusResponse(BaseModel):
    status: str
    documents_indexed: int
    model: str
    embedding_model: str
