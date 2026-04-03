"""Document ingestion API routes."""

from typing import List

from fastapi import APIRouter, Depends, File, UploadFile

from api.dependencies import get_ingestion_service
from core.schemas import IngestionResponse
from ingestion.service import IngestionService

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("", response_model=IngestionResponse)
async def ingest_documents(
    files: List[UploadFile] = File(...),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResponse:
    """Ingest uploaded PDF, TXT, and Markdown files into the vector store."""
    return await ingestion_service.ingest_files(files)
