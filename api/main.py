"""FastAPI application entrypoint."""

from fastapi import FastAPI

from api.routes_feedback import router as feedback_router
from api.routes_ingestion import router as ingestion_router
from api.routes_query import router as query_router
from api.routes_tracking import router as tracking_router
from core.database import SQLiteManager
from core.config import get_settings

settings = get_settings()
SQLiteManager(settings)

app = FastAPI(
    title="Production RAG System",
    description="Production-ready RAG service with hallucination detection, feedback loop, and MLflow tracking.",
    version="1.0.0",
)

app.include_router(ingestion_router)
app.include_router(query_router)
app.include_router(feedback_router)
app.include_router(tracking_router)


@app.get("/health")
async def health() -> dict:
    """Basic health endpoint."""
    return {"status": "ok", "environment": settings.environment}
