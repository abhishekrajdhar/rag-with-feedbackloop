"""Application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings for the RAG service."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = Field(default="development")
    llm_provider: str = Field(default="openai")
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-1.5-flash")
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    hallucination_model_name: str = Field(default="cross-encoder/nli-deberta-v3-small")
    chroma_collection_name: str = Field(default="rag_documents")
    chroma_persist_directory: str = Field(default="data/chroma")
    uploads_directory: str = Field(default="data/uploads")
    sqlite_db_path: str = Field(default="data/rag_app.db")
    mlflow_tracking_uri: str = Field(default="file:./mlruns")
    mlflow_experiment_name: str = Field(default="rag_production")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    default_top_k: int = Field(default=4)
    max_top_k: int = Field(default=10)
    llm_temperature: float = Field(default=0.1)
    request_timeout_seconds: int = Field(default=60)
    hallucination_threshold: float = Field(default=0.45)

    @property
    def sqlite_db_abspath(self) -> Path:
        """Return the SQLite database path as an absolute path."""
        return Path(self.sqlite_db_path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
