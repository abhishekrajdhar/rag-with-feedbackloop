"""Utility helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List
from uuid import uuid4


def utc_now_iso() -> str:
    """Return the current UTC timestamp as ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def generate_query_id() -> str:
    """Generate a unique query id."""
    return str(uuid4())


def coerce_text(value: str) -> str:
    """Normalize text for storage and prompting."""
    return " ".join(value.split())


def format_sse(event: str, data: str) -> str:
    """Format a Server-Sent Events message."""
    return f"event: {event}\ndata: {data}\n\n"


def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    """Yield a list in fixed-size batches."""
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]
