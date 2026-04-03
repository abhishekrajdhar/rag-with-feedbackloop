"""MLflow summary routes."""

from fastapi import APIRouter, Depends

from api.dependencies import get_tracking_service
from tracking.service import TrackingService

router = APIRouter(prefix="/mlflow", tags=["tracking"])


@router.get("/summary")
async def mlflow_summary(
    tracking_service: TrackingService = Depends(get_tracking_service),
) -> dict:
    """Return a compact summary of tracked query runs."""
    return tracking_service.get_summary()
