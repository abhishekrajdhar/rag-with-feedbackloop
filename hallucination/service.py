"""Hallucination detection using an NLI cross-encoder."""

from __future__ import annotations

from typing import Iterable, List

import torch
from sentence_transformers import CrossEncoder

from core.config import Settings
from core.models import Verdict


class HallucinationService:
    """Compute entailment-based support scores for generated answers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = CrossEncoder(settings.hallucination_model_name)

    def score_answer(self, answer: str, contexts: Iterable[str]) -> float:
        """Return the strongest entailment score across retrieved contexts."""
        pairs = [(context, answer) for context in contexts if context.strip()]
        if not pairs:
            return 0.0

        logits = self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        tensor = torch.tensor(logits)
        probabilities = torch.softmax(tensor, dim=1).numpy()
        entailment_scores: List[float] = [float(row[-1]) for row in probabilities]
        return round(max(entailment_scores), 4)

    def verdict_for_score(self, score: float) -> Verdict:
        """Convert a numeric score into a hallucination verdict."""
        if score < self.settings.hallucination_threshold:
            return Verdict.HALLUCINATED
        return Verdict.SAFE
