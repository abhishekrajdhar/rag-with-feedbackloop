"""Query API routes."""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_rag_service
from core.schemas import QueryRequest, QueryResponse
from retrieval.service import RAGService

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    payload: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """Retrieve relevant chunks, generate an answer, and detect hallucinations."""
    return await rag_service.answer_question(payload)


@router.post("/query/stream")
async def stream_query(
    payload: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    """Stream an answer as Server-Sent Events."""
    generator = rag_service.stream_answer(payload)
    return StreamingResponse(generator, media_type="text/event-stream")
