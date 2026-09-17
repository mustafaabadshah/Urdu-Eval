"""BLEU score metric implementation."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from urdu_eval.metrics.base import Metric, register_metric
from urdu_eval.metrics.f1 import tokenize_text
from urdu_eval.models import MetricResult


def get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a list of tokens."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def calculate_bleu(
    pred_tokens: list[str],
    ref_tokens_list: list[list[str]],
    max_order: int = 4,
    smooth: bool = True,
) -> tuple[float, list[float], float]:
    """Calculate BLEU score with modified n-gram precision and brevity penalty."""
    if not pred_tokens:
        return 0.0, [0.0] * max_order, 0.0

    # Brevity penalty
    pred_len = len(pred_tokens)
    ref_lens = [len(ref) for ref in ref_tokens_list]
    # Closest reference length
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - pred_len), ref_len))

    if pred_len > closest_ref_len:
        bp = 1.0
    elif pred_len > 0:
        bp = math.exp(1.0 - closest_ref_len / pred_len)
    else:
        bp = 0.0

    precisions: list[float] = []
    for order in range(1, max_order + 1):
        pred_ngrams = get_ngrams(pred_tokens, order)
        if not pred_ngrams:
            precisions.append(0.0)
            continue

        pred_counts = Counter(pred_ngrams)
        max_ref_counts: dict[tuple[str, ...], int] = {}
        for ref_tokens in ref_tokens_list:
            ref_ngrams = get_ngrams(ref_tokens, order)
            ref_counts = Counter(ref_ngrams)
            for ng in ref_counts:
                max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), ref_counts[ng])

        clipped_count = 0
        total_count = sum(pred_counts.values())
        for ng, count in pred_counts.items():
            clipped_count += min(count, max_ref_counts.get(ng, 0))

        if total_count > 0 and clipped_count > 0:
            precisions.append(clipped_count / total_count)
        elif smooth:
            # Add-1 smoothing for higher orders if earlier orders had matches
            precisions.append(1.0 / (2.0 * total_count) if total_count > 0 else 0.0)
        else:
            precisions.append(0.0)

    if min(precisions) > 0.0:
        score = bp * math.exp(sum(math.log(p) for p in precisions) / max_order)
    else:
        score = 0.0

    return score, precisions, bp


@register_metric("bleu")
class BLEUMetric(Metric):
    """BLEU translation metric (standard 4-gram with brevity penalty)."""

    name = "bleu"

    def __init__(self, max_order: int = 4, smooth: bool = True) -> None:
        self.max_order = max_order
        self.smooth = smooth

    def compute(self, prediction: str, reference: str | list[str], **kwargs: Any) -> float:
        pred_tokens = tokenize_text(prediction)
        refs = [reference] if isinstance(reference, str) else reference
        ref_tokens_list = [tokenize_text(ref) for ref in refs]

        score, _, _ = calculate_bleu(
            pred_tokens, ref_tokens_list, max_order=self.max_order, smooth=self.smooth
        )
        return score

    def compute_details(
        self, prediction: str, reference: str | list[str], **kwargs: Any
    ) -> MetricResult:
        pred_tokens = tokenize_text(prediction)
        refs = [reference] if isinstance(reference, str) else reference
        ref_tokens_list = [tokenize_text(ref) for ref in refs]

        score, precisions, bp = calculate_bleu(
            pred_tokens, ref_tokens_list, max_order=self.max_order, smooth=self.smooth
        )
        return MetricResult(
            metric_name=self.name,
            score=score,
            details={
                "precisions": precisions,
                "brevity_penalty": bp,
            },
        )
