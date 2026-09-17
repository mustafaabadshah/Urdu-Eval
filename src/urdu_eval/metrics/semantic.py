"""Semantic similarity metric."""

from __future__ import annotations

import difflib
from typing import Any

from urdu_eval.metrics.base import Metric, register_metric
from urdu_eval.metrics.f1 import tokenize_text
from urdu_eval.models import MetricResult


def token_jaccard_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


@register_metric("semantic_similarity")
@register_metric("semantic")
class SemanticSimilarityMetric(Metric):
    """Semantic and lexical sequence similarity metric."""

    name = "semantic_similarity"

    def compute(self, prediction: str, reference: str | list[str], **kwargs: Any) -> float:
        refs = [reference] if isinstance(reference, str) else reference
        best_score = 0.0

        for ref in refs:
            # SequenceMatcher ratio (character-level difflib)
            char_sim = difflib.SequenceMatcher(None, prediction.strip(), ref.strip()).ratio()

            # Word-level Jaccard
            pred_tokens = tokenize_text(prediction)
            ref_tokens = tokenize_text(ref)
            word_sim = token_jaccard_similarity(pred_tokens, ref_tokens)

            combined = (0.5 * char_sim) + (0.5 * word_sim)
            if combined > best_score:
                best_score = combined

        return best_score

    def compute_details(
        self, prediction: str, reference: str | list[str], **kwargs: Any
    ) -> MetricResult:
        score = self.compute(prediction, reference, **kwargs)
        return MetricResult(
            metric_name=self.name,
            score=score,
            details={"similarity": score},
        )
