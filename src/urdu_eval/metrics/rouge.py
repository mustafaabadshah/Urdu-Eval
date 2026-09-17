"""ROUGE metric implementations (ROUGE-1, ROUGE-2, ROUGE-L)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from urdu_eval.metrics.base import Metric, register_metric
from urdu_eval.metrics.f1 import tokenize_text
from urdu_eval.models import MetricResult


def compute_ngram_overlap(
    pred_tokens: list[str], ref_tokens: list[str], n: int
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 for word n-grams."""
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0, 0.0, 0.0

    pred_ngrams = [tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)]
    ref_ngrams = [tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)]

    pred_counts = Counter(pred_ngrams)
    ref_counts = Counter(ref_ngrams)
    overlap = sum((pred_counts & ref_counts).values())

    precision = overlap / len(pred_ngrams) if pred_ngrams else 0.0
    recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of Longest Common Subsequence using dynamic programming."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0

    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_rouge_l(pred_tokens: list[str], ref_tokens: list[str]) -> tuple[float, float, float]:
    """Compute ROUGE-L precision, recall, and F1."""
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


@register_metric("rouge1")
@register_metric("rouge-1")
class ROUGE1Metric(Metric):
    """ROUGE-1 unigram overlap metric."""

    name = "rouge-1"

    def compute(self, prediction: str, reference: str | list[str], **kwargs: Any) -> float:
        pred_tokens = tokenize_text(prediction)
        refs = [reference] if isinstance(reference, str) else reference
        best_f1 = 0.0

        for ref in refs:
            ref_tokens = tokenize_text(ref)
            _, _, f1 = compute_ngram_overlap(pred_tokens, ref_tokens, n=1)
            if f1 > best_f1:
                best_f1 = f1

        return best_f1

    def compute_details(
        self, prediction: str, reference: str | list[str], **kwargs: Any
    ) -> MetricResult:
        score = self.compute(prediction, reference, **kwargs)
        return MetricResult(metric_name=self.name, score=score)


@register_metric("rouge2")
@register_metric("rouge-2")
class ROUGE2Metric(Metric):
    """ROUGE-2 bigram overlap metric."""

    name = "rouge-2"

    def compute(self, prediction: str, reference: str | list[str], **kwargs: Any) -> float:
        pred_tokens = tokenize_text(prediction)
        refs = [reference] if isinstance(reference, str) else reference
        best_f1 = 0.0

        for ref in refs:
            ref_tokens = tokenize_text(ref)
            _, _, f1 = compute_ngram_overlap(pred_tokens, ref_tokens, n=2)
            if f1 > best_f1:
                best_f1 = f1

        return best_f1

    def compute_details(
        self, prediction: str, reference: str | list[str], **kwargs: Any
    ) -> MetricResult:
        score = self.compute(prediction, reference, **kwargs)
        return MetricResult(metric_name=self.name, score=score)


@register_metric("rougel")
@register_metric("rouge-l")
class ROUGELMetric(Metric):
    """ROUGE-L Longest Common Subsequence metric."""

    name = "rouge-l"

    def compute(self, prediction: str, reference: str | list[str], **kwargs: Any) -> float:
        pred_tokens = tokenize_text(prediction)
        refs = [reference] if isinstance(reference, str) else reference
        best_f1 = 0.0

        for ref in refs:
            ref_tokens = tokenize_text(ref)
            _, _, f1 = compute_rouge_l(pred_tokens, ref_tokens)
            if f1 > best_f1:
                best_f1 = f1

        return best_f1

    def compute_details(
        self, prediction: str, reference: str | list[str], **kwargs: Any
    ) -> MetricResult:
        score = self.compute(prediction, reference, **kwargs)
        return MetricResult(metric_name=self.name, score=score)
