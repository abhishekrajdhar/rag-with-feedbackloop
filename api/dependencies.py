"""FastAPI dependency providers."""

from functools import lru_cache

from core.config import Settings, get_settings
from core.database import SQLiteManager
from core.llm import LLMService
from core.vector_store import VectorStoreManager
from feedback.service import FeedbackService
from hallucination.service import HallucinationService
from ingestion.service import IngestionService
from retrieval.service import RAGService
from tracking.service import TrackingService


@lru_cache(maxsize=1)
def get_database() -> SQLiteManager:
    """Return the shared SQLite manager."""
    return SQLiteManager(get_settings())


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreManager:
    """Return the shared vector store manager."""
    return VectorStoreManager(get_settings())


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """Return the shared LLM service."""
    return LLMService(get_settings())


@lru_cache(maxsize=1)
def get_hallucination_service() -> HallucinationService:
    """Return the shared hallucination detection service."""
    return HallucinationService(get_settings())


@lru_cache(maxsize=1)
def get_tracking_service() -> TrackingService:
    """Return the shared MLflow tracking service."""
    return TrackingService(get_settings())


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    """Return the shared ingestion service."""
    return IngestionService(get_settings(), get_vector_store())


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Return the shared RAG orchestration service."""
    return RAGService(
        settings=get_settings(),
        vector_store=get_vector_store(),
        llm_service=get_llm_service(),
        hallucination_service=get_hallucination_service(),
        database=get_database(),
        tracking_service=get_tracking_service(),
    )


@lru_cache(maxsize=1)
def get_feedback_service() -> FeedbackService:
    """Return the shared feedback service."""
    return FeedbackService(
        settings=get_settings(),
        database=get_database(),
        vector_store=get_vector_store(),
        rag_service=get_rag_service(),
    )


def settings_dependency() -> Settings:
    """FastAPI dependency wrapper around cached settings."""
    return get_settings()
