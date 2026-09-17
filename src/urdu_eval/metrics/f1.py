"""Token F1 and SQuAD F1 metrics."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from urdu_eval.metrics.base import Metric, register_metric
from urdu_eval.models import MetricResult

# Punctuation regex including Urdu punctuation (، ؛ ؟ ۔)
PUNCTUATION_REGEX = re.compile(r"[\.,!?:;\-\–\—\(\)\[\]\{\}\"\'`\u060C\u061B\u061F\u06D4]")


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase words stripped of punctuation."""
    clean = PUNCTUATION_REGEX.sub(" ", text).lower()
    return clean.split()


def compute_f1_score(pred_tokens: list[str], ref_tokens: list[str]) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 between two token lists."""
    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())

    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 2.0 * (precision * recall) / (precision + recall)
    return precision, recall, f1


@register_metric("f1")
@register_metric("macro_f1")
class F1Metric(Metric):
    """Token-level F1 metric."""

    name = "f1"

    def compute(self, prediction: str, reference: str | list[str], **kwargs: Any) -> float:
        pred_tokens = tokenize_text(prediction)
        refs = [reference] if isinstance(reference, str) else reference

        best_f1 = 0.0
        for ref in refs:
            ref_tokens = tokenize_text(ref)
            _, _, f1 = compute_f1_score(pred_tokens, ref_tokens)
            if f1 > best_f1:
                best_f1 = f1

        return best_f1

    def compute_details(
        self, prediction: str, reference: str | list[str], **kwargs: Any
    ) -> MetricResult:
        pred_tokens = tokenize_text(prediction)
        refs = [reference] if isinstance(reference, str) else reference

        best_f1 = 0.0
        best_p = 0.0
        best_r = 0.0

        for ref in refs:
            ref_tokens = tokenize_text(ref)
            p, r, f1 = compute_f1_score(pred_tokens, ref_tokens)
            if f1 >= best_f1:
                best_f1 = f1
                best_p = p
                best_r = r

        return MetricResult(
            metric_name=self.name,
            score=best_f1,
            details={"precision": best_p, "recall": best_r, "f1": best_f1},
        )


@register_metric("squad_f1")
class SquadF1Metric(F1Metric):
    """SQuAD-style QA F1 metric (alias for F1Metric)."""

    name = "squad_f1"
