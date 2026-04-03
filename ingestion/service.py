"""Document ingestion logic."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from core.config import Settings
from core.schemas import IngestionFileStat, IngestionResponse
from core.utils import utc_now_iso
from core.vector_store import VectorStoreManager


class IngestionService:
    """Loads files, chunks content, and stores embeddings."""

    allowed_suffixes = {".pdf", ".txt", ".md"}

    def __init__(self, settings: Settings, vector_store: VectorStoreManager) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        Path(settings.uploads_directory).mkdir(parents=True, exist_ok=True)

    async def ingest_files(self, files: List[UploadFile]) -> IngestionResponse:
        """Ingest uploaded files into the vector store."""
        if not files:
            raise HTTPException(status_code=400, detail="At least one file is required.")

        documents: List[Document] = []
        ids: List[str] = []
        stats: List[IngestionFileStat] = []
        timestamp = utc_now_iso()

        for file in files:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in self.allowed_suffixes:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

            raw_bytes = await file.read()
            file_token = uuid4().hex[:8]
            original_name = file.filename or "uploaded_file"
            source_path = Path(self.settings.uploads_directory) / f"{file_token}_{original_name}"
            source_path.write_bytes(raw_bytes)

            raw_text = self._extract_text(suffix=suffix, raw_bytes=raw_bytes)
            if not raw_text.strip():
                raise HTTPException(status_code=400, detail=f"No text extracted from {file.filename}")

            chunks = self.text_splitter.split_text(raw_text)
            stats.append(IngestionFileStat(filename=file.filename or "unknown", chunks=len(chunks)))

            for chunk_index, chunk in enumerate(chunks):
                chunk_id = f"{Path(original_name).stem}-{file_token}-{chunk_index}"
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source_file": original_name,
                            "chunk_id": chunk_id,
                            "timestamp": timestamp,
                            "priority": 0,
                        },
                    )
                )
                ids.append(chunk_id)

        self.vector_store.add_documents(documents, ids)
        return IngestionResponse(
            files_processed=len(stats),
            chunks_indexed=len(ids),
            collection_name=self.settings.chroma_collection_name,
            files=stats,
        )

    def _extract_text(self, suffix: str, raw_bytes: bytes) -> str:
        """Extract plaintext from supported file types."""
        if suffix in {".txt", ".md"}:
            return raw_bytes.decode("utf-8", errors="ignore")
        if suffix == ".pdf":
            reader = PdfReader(BytesIO(raw_bytes))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        raise ValueError(f"Unsupported suffix: {suffix}")
