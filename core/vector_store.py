"""Vector store abstraction built on persistent Chroma."""

from __future__ import annotations

from typing import Dict, List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from core.config import Settings


class VectorStoreManager:
    """Manages embeddings and Chroma persistence."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
        self.vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_directory,
        )

    def add_documents(self, documents: List[Document], ids: List[str]) -> None:
        """Add documents to the collection."""
        self.vector_store.add_documents(documents=documents, ids=ids)

    def add_texts(self, texts: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        """Add plain text chunks to the collection."""
        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def similarity_search(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve documents and normalize scores into a 0..1 relevance scale."""
        pairs = self.vector_store.similarity_search_with_score(query=query, k=max(top_k * 2, top_k))
        ranked_results: List[Dict] = []
        for document, distance in pairs:
            metadata = document.metadata or {}
            priority = int(metadata.get("priority", 0))
            relevance = 1.0 / (1.0 + float(distance))
            boosted_score = min(relevance + (0.12 * priority), 1.0)
            ranked_results.append(
                {
                    "document": document,
                    "score": round(boosted_score, 4),
                    "distance": float(distance),
                }
            )

        ranked_results.sort(key=lambda item: item["score"], reverse=True)
        return ranked_results[:top_k]
