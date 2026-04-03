"""Feedback loop implementation."""

from __future__ import annotations

from fastapi import HTTPException
from langchain_core.documents import Document

from core.config import Settings
from core.database import SQLiteManager
from core.schemas import FeedbackRequest, FeedbackResponse, QueryRequest
from core.utils import utc_now_iso
from core.vector_store import VectorStoreManager
from retrieval.service import RAGService


class FeedbackService:
    """Persists user feedback and injects corrective memory into retrieval."""

    def __init__(
        self,
        settings: Settings,
        database: SQLiteManager,
        vector_store: VectorStoreManager,
        rag_service: RAGService,
    ) -> None:
        self.settings = settings
        self.database = database
        self.vector_store = vector_store
        self.rag_service = rag_service

    async def process_feedback(self, payload: FeedbackRequest) -> FeedbackResponse:
        """Persist feedback and optionally re-run the query with correction memory."""
        existing_query = self.database.get_query(payload.query_id)
        if existing_query is None:
            raise HTTPException(status_code=404, detail=f"Query '{payload.query_id}' was not found.")

        created_at = utc_now_iso()
        feedback_id = self.database.insert_feedback(
            query_id=payload.query_id,
            rating=payload.rating,
            correction=payload.correction,
            created_at=created_at,
        )

        correction_ingested = False
        improved_answer = None
        improved_query_id = None

        if payload.rating <= 2 and payload.correction:
            correction_ingested = True
            chunk_id = f"feedback-{feedback_id}"
            correction_text = (
                f"User correction for question: {existing_query['question']}\n"
                f"Correction: {payload.correction}"
            )
            document = Document(
                page_content=correction_text,
                metadata={
                    "source_file": "user_feedback",
                    "chunk_id": chunk_id,
                    "timestamp": created_at,
                    "priority": 2,
                    "feedback_for_query_id": payload.query_id,
                },
            )
            self.vector_store.add_documents([document], [chunk_id])

            improved = await self.rag_service.answer_question(
                QueryRequest(question=existing_query["question"], top_k=existing_query["top_k"])
            )
            improved_answer = improved.answer
            improved_query_id = improved.query_id

        return FeedbackResponse(
            feedback_id=feedback_id,
            query_id=payload.query_id,
            old_answer=existing_query["answer"],
            improved_answer=improved_answer,
            improved_query_id=improved_query_id,
            correction_ingested=correction_ingested,
        )
