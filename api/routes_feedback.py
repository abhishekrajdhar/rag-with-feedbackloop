"""Feedback API routes."""

from fastapi import APIRouter, Depends

from api.dependencies import get_feedback_service
from core.schemas import FeedbackRequest, FeedbackResponse
from feedback.service import FeedbackService

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    payload: FeedbackRequest,
    feedback_service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackResponse:
    """Persist feedback and optionally inject corrective knowledge."""
    return await feedback_service.process_feedback(payload)
