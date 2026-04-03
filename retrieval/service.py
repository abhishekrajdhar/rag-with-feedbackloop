"""Retrieval and answer generation orchestration."""

from __future__ import annotations

import json
import time
from typing import AsyncGenerator, Dict, List, Tuple

from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.config import Settings
from core.database import SQLiteManager
from core.llm import LLMService
from core.models import Verdict
from core.schemas import QueryRequest, QueryResponse, RetrievalScoreItem, SourceItem
from core.utils import format_sse, generate_query_id, utc_now_iso
from core.vector_store import VectorStoreManager
from hallucination.service import HallucinationService
from tracking.service import TrackingService

BASE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a retrieval-grounded assistant.
Answer ONLY from the supplied context.
If the context is insufficient, say: "I do not have enough information in the retrieved documents to answer that safely."
Do not invent facts, citations, or external knowledge.
Keep the answer concise and factual.
            """.strip(),
        ),
        (
            "human",
            """
Question:
{question}

Retrieved context:
{context}

Return a grounded answer and mention the supporting source_file names when possible.
            """.strip(),
        ),
    ]
)

STRICT_RETRY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Your previous answer was flagged as insufficiently supported.
Rewrite the answer using ONLY exact facts that appear in the context.
If any claim cannot be fully supported, omit it.
If the context is insufficient, say: "I do not have enough information in the retrieved documents to answer that safely."
            """.strip(),
        ),
        (
            "human",
            """
Question:
{question}

Retrieved context:
{context}
            """.strip(),
        ),
    ]
)


class RAGService:
    """Runs retrieval, generation, hallucination checks, storage, and tracking."""

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStoreManager,
        llm_service: LLMService,
        hallucination_service: HallucinationService,
        database: SQLiteManager,
        tracking_service: TrackingService,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.llm = llm_service.client
        self.hallucination_service = hallucination_service
        self.database = database
        self.tracking_service = tracking_service
        self.output_parser = StrOutputParser()

    async def answer_question(self, payload: QueryRequest) -> QueryResponse:
        """Retrieve context, generate an answer, and apply hallucination detection."""
        started = time.perf_counter()
        top_k = min(payload.top_k or self.settings.default_top_k, self.settings.max_top_k)
        retrieval_results = self.vector_store.similarity_search(payload.question, top_k)
        if not retrieval_results:
            raise HTTPException(status_code=404, detail="No documents found. Ingest documents first.")

        sources, retrieval_scores, context = self._prepare_context(retrieval_results)
        answer = await self._generate_answer(payload.question, context, retry=False)
        hallucination_score = self.hallucination_service.score_answer(answer, [item["document"].page_content for item in retrieval_results])
        verdict = self.hallucination_service.verdict_for_score(hallucination_score)
        retried = False

        if verdict == Verdict.HALLUCINATED:
            retried = True
            answer = await self._generate_answer(payload.question, context, retry=True)
            hallucination_score = self.hallucination_service.score_answer(
                answer, [item["document"].page_content for item in retrieval_results]
            )
            verdict = self.hallucination_service.verdict_for_score(hallucination_score)
            if verdict == Verdict.HALLUCINATED:
                answer = (
                    "Disclaimer: this answer may be partially unsupported by the retrieved context.\n\n"
                    f"{answer}"
                )

        latency = round(time.perf_counter() - started, 4)
        query_id = generate_query_id()
        created_at = utc_now_iso()

        response = QueryResponse(
            query_id=query_id,
            answer=answer,
            sources=sources,
            retrieval_scores=retrieval_scores,
            latency=latency,
            hallucination_score=hallucination_score,
            verdict=verdict.value,
            retried=retried,
        )
        self._persist_result(
            query_id=query_id,
            question=payload.question,
            response=response,
            top_k=top_k,
            created_at=created_at,
        )
        return response

    async def stream_answer(self, payload: QueryRequest) -> AsyncGenerator[str, None]:
        """Stream a query result over Server-Sent Events."""
        started = time.perf_counter()
        top_k = min(payload.top_k or self.settings.default_top_k, self.settings.max_top_k)
        retrieval_results = self.vector_store.similarity_search(payload.question, top_k)
        if not retrieval_results:
            yield format_sse("error", json.dumps({"detail": "No documents found. Ingest documents first."}))
            return

        sources, retrieval_scores, context = self._prepare_context(retrieval_results)
        yield format_sse(
            "metadata",
            json.dumps(
                {
                    "sources": [source.model_dump() for source in sources],
                    "retrieval_scores": [score.model_dump() for score in retrieval_scores],
                }
            ),
        )

        prompt = BASE_PROMPT.format_messages(question=payload.question, context=context)
        chunks: List[str] = []
        async for chunk in self.llm.astream(prompt):
            text = getattr(chunk, "content", "")
            if not text:
                continue
            chunks.append(text)
            yield format_sse("token", json.dumps({"text": text}))

        answer = "".join(chunks).strip()
        hallucination_score = self.hallucination_service.score_answer(
            answer, [item["document"].page_content for item in retrieval_results]
        )
        verdict = self.hallucination_service.verdict_for_score(hallucination_score)
        retried = False
        if verdict == Verdict.HALLUCINATED:
            retried = True
            answer = await self._generate_answer(payload.question, context, retry=True)
            hallucination_score = self.hallucination_service.score_answer(
                answer, [item["document"].page_content for item in retrieval_results]
            )
            verdict = self.hallucination_service.verdict_for_score(hallucination_score)

        latency = round(time.perf_counter() - started, 4)
        query_id = generate_query_id()
        created_at = utc_now_iso()
        response = QueryResponse(
            query_id=query_id,
            answer=answer,
            sources=sources,
            retrieval_scores=retrieval_scores,
            latency=latency,
            hallucination_score=hallucination_score,
            verdict=verdict.value,
            retried=retried,
        )
        self._persist_result(
            query_id=query_id,
            question=payload.question,
            response=response,
            top_k=top_k,
            created_at=created_at,
        )
        yield format_sse("done", response.model_dump_json())

    def _prepare_context(
        self, retrieval_results: List[Dict]
    ) -> Tuple[List[SourceItem], List[RetrievalScoreItem], str]:
        """Convert retrieved documents into API payloads and prompt context."""
        sources: List[SourceItem] = []
        retrieval_scores: List[RetrievalScoreItem] = []
        context_parts: List[str] = []

        for index, result in enumerate(retrieval_results, start=1):
            document: Document = result["document"]
            metadata = document.metadata or {}
            source = SourceItem(
                source_file=str(metadata.get("source_file", "unknown")),
                chunk_id=str(metadata.get("chunk_id", f"chunk-{index}")),
                content=document.page_content,
                timestamp=str(metadata.get("timestamp", "")),
                priority=int(metadata.get("priority", 0)),
            )
            sources.append(source)
            retrieval_scores.append(
                RetrievalScoreItem(chunk_id=source.chunk_id, score=round(float(result["score"]), 4))
            )
            context_parts.append(
                f"[{source.chunk_id}] source_file={source.source_file}\n{document.page_content}"
            )

        return sources, retrieval_scores, "\n\n".join(context_parts)

    async def _generate_answer(self, question: str, context: str, retry: bool) -> str:
        """Generate an answer from the chosen prompt."""
        prompt = STRICT_RETRY_PROMPT if retry else BASE_PROMPT
        chain = prompt | self.llm | self.output_parser
        return (await chain.ainvoke({"question": question, "context": context})).strip()

    def _persist_result(
        self,
        query_id: str,
        question: str,
        response: QueryResponse,
        top_k: int,
        created_at: str,
    ) -> None:
        """Persist logs and track the run in MLflow."""
        self.database.insert_query_log(
            {
                "id": query_id,
                "question": question,
                "answer": response.answer,
                "sources": [item.model_dump() for item in response.sources],
                "retrieval_scores": [item.model_dump() for item in response.retrieval_scores],
                "latency": response.latency,
                "top_k": top_k,
                "hallucination_score": response.hallucination_score,
                "verdict": response.verdict,
                "retried": response.retried,
                "created_at": created_at,
            }
        )
        self.database.insert_hallucination_log(
            {
                "query_id": query_id,
                "query": question,
                "answer": response.answer,
                "score": response.hallucination_score,
                "verdict": response.verdict,
                "timestamp": created_at,
            }
        )
        self.tracking_service.log_query_run(
            query_id=query_id,
            question=question,
            top_k=top_k,
            latency=response.latency,
            hallucination_score=response.hallucination_score,
            verdict=response.verdict,
            retried=response.retried,
        )
