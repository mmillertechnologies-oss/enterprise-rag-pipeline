from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "documents"

    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_length: int = 50

    retrieval_k: int = 5
    retrieval_score_threshold: float = 0.3

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
