"""chrF and chrF++ metric implementations."""

from __future__ import annotations

from collections import Counter
from typing import Any

from urdu_eval.metrics.base import Metric, register_metric
from urdu_eval.metrics.f1 import tokenize_text
from urdu_eval.models import MetricResult


def get_char_ngrams(text: str, n: int) -> list[str]:
    """Extract character n-grams (ignoring extra whitespace)."""
    collapsed = "".join(text.split())
    if len(collapsed) < n:
        return []
    return [collapsed[i : i + n] for i in range(len(collapsed) - n + 1)]


def get_word_ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract word n-grams."""
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def calculate_chrf_score(
    prediction: str,
    reference: str,
    char_order: int = 6,
    word_order: int = 2,
    beta: float = 2.0,
) -> float:
    """Calculate chrF++ score between prediction and reference."""
    precisions: list[float] = []
    recalls: list[float] = []

    # Character n-grams (1 to char_order)
    for n in range(1, char_order + 1):
        pred_ngrams = Counter(get_char_ngrams(prediction, n))
        ref_ngrams = Counter(get_char_ngrams(reference, n))

        overlap = sum((pred_ngrams & ref_ngrams).values())
        pred_total = sum(pred_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        p = overlap / pred_total if pred_total > 0 else 0.0
        r = overlap / ref_total if ref_total > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

    # Word n-grams (1 to word_order) if word_order > 0 (chrF++)
    if word_order > 0:
        pred_words = tokenize_text(prediction)
        ref_words = tokenize_text(reference)

        for n in range(1, word_order + 1):
            pred_w_ngrams = Counter(get_word_ngrams(pred_words, n))
            ref_w_ngrams = Counter(get_word_ngrams(ref_words, n))

            overlap = sum((pred_w_ngrams & ref_w_ngrams).values())
            pred_total = sum(pred_w_ngrams.values())
            ref_total = sum(ref_w_ngrams.values())

            p = overlap / pred_total if pred_total > 0 else 0.0
            r = overlap / ref_total if ref_total > 0 else 0.0
            precisions.append(p)
            recalls.append(r)

    # Average precision and recall across all orders
    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0

    beta_sq = beta**2
    if avg_p + avg_r == 0.0:
        return 0.0

    # F_beta formula
    f_score = (1.0 + beta_sq) * (avg_p * avg_r) / ((beta_sq * avg_p) + avg_r)
    return f_score


@register_metric("chrf")
@register_metric("chrf++")
class ChrFMetric(Metric):
    """chrF++ character and word n-gram F-score."""

    name = "chrf++"

    def __init__(
        self,
        char_order: int = 6,
        word_order: int = 2,
        beta: float = 2.0,
    ) -> None:
        self.char_order = char_order
        self.word_order = word_order
        self.beta = beta

    def compute(self, prediction: str, reference: str | list[str], **kwargs: Any) -> float:
        refs = [reference] if isinstance(reference, str) else reference
        best_score = 0.0

        for ref in refs:
            score = calculate_chrf_score(
                prediction=prediction,
                reference=ref,
                char_order=self.char_order,
                word_order=self.word_order,
                beta=self.beta,
            )
            if score > best_score:
                best_score = score

        return best_score

    def compute_details(
        self, prediction: str, reference: str | list[str], **kwargs: Any
    ) -> MetricResult:
        score = self.compute(prediction, reference, **kwargs)
        return MetricResult(
            metric_name=self.name,
            score=score,
            details={
                "char_order": self.char_order,
                "word_order": self.word_order,
                "beta": self.beta,
            },
        )
