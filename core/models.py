"""Shared application models."""

from enum import Enum


class Verdict(str, Enum):
    """Hallucination verdicts."""

    SAFE = "SAFE"
    HALLUCINATED = "HALLUCINATED"
