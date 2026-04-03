"""SQLite persistence layer."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from core.config import Settings


class SQLiteManager:
    """Simple SQLite manager for query, hallucination, and feedback logs."""

    def __init__(self, settings: Settings) -> None:
        self.db_path = str(settings.sqlite_db_abspath)
        settings.sqlite_db_abspath.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a connection with row factory configured."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS queries (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    retrieval_scores_json TEXT NOT NULL,
                    latency REAL NOT NULL,
                    top_k INTEGER NOT NULL,
                    hallucination_score REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    retried INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS hallucination_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    score REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    correction TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def insert_query_log(self, payload: Dict[str, Any]) -> None:
        """Persist a completed query result."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO queries (
                    id, question, answer, sources_json, retrieval_scores_json, latency,
                    top_k, hallucination_score, verdict, retried, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload["question"],
                    payload["answer"],
                    json.dumps(payload["sources"]),
                    json.dumps(payload["retrieval_scores"]),
                    payload["latency"],
                    payload["top_k"],
                    payload["hallucination_score"],
                    payload["verdict"],
                    int(payload["retried"]),
                    payload["created_at"],
                ),
            )

    def insert_hallucination_log(self, payload: Dict[str, Any]) -> None:
        """Persist hallucination detection result."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO hallucination_log (query_id, query, answer, score, verdict, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["query_id"],
                    payload["query"],
                    payload["answer"],
                    payload["score"],
                    payload["verdict"],
                    payload["timestamp"],
                ),
            )

    def insert_feedback(self, query_id: str, rating: int, correction: Optional[str], created_at: str) -> int:
        """Persist user feedback and return the new feedback id."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback (query_id, rating, correction, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (query_id, rating, correction, created_at),
            )
            return int(cursor.lastrowid)

    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a query record by id."""
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM queries WHERE id = ?", (query_id,)).fetchone()
            if row is None:
                return None
            return {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "sources": json.loads(row["sources_json"]),
                "retrieval_scores": json.loads(row["retrieval_scores_json"]),
                "latency": row["latency"],
                "top_k": row["top_k"],
                "hallucination_score": row["hallucination_score"],
                "verdict": row["verdict"],
                "retried": bool(row["retried"]),
                "created_at": row["created_at"],
            }
