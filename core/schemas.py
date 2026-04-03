"""Pydantic schemas for API requests and responses."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class IngestionFileStat(BaseModel):
    """Per-file ingestion summary."""

    filename: str
    chunks: int


class IngestionResponse(BaseModel):
    """Response returned by the ingestion endpoint."""

    files_processed: int
    chunks_indexed: int
    collection_name: str
    files: List[IngestionFileStat]


class QueryRequest(BaseModel):
    """Incoming query payload."""

    question: str = Field(..., min_length=3)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceItem(BaseModel):
    """Retrieved source metadata."""

    source_file: str
    chunk_id: str
    content: str
    timestamp: str
    priority: int = 0


class RetrievalScoreItem(BaseModel):
    """Similarity score for a retrieved chunk."""

    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    """Query response payload."""

    query_id: str
    answer: str
    sources: List[SourceItem]
    retrieval_scores: List[RetrievalScoreItem]
    latency: float
    hallucination_score: float
    verdict: str
    retried: bool


class FeedbackRequest(BaseModel):
    """Feedback submission payload."""

    query_id: str
    rating: int = Field(..., ge=1, le=5)
    correction: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("correction")
    @classmethod
    def normalize_correction(cls, value: Optional[str]) -> Optional[str]:
        """Strip empty correction payloads."""
        if value is None:
            return None
        value = value.strip()
        return value or None


class FeedbackResponse(BaseModel):
    """Feedback processing result."""

    feedback_id: int
    query_id: str
    old_answer: str
    improved_answer: Optional[str] = None
    improved_query_id: Optional[str] = None
    correction_ingested: bool
