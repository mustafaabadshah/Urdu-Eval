"""Exact match and accuracy metric implementations."""

from __future__ import annotations

from typing import Any

from urdu_eval.metrics.base import Metric, register_metric
from urdu_eval.models import MetricResult


@register_metric("exact_match")
@register_metric("accuracy")
class ExactMatchMetric(Metric):
    """Exact match scoring: 1.0 if prediction matches reference, else 0.0."""

    name = "exact_match"

    def __init__(self, ignore_case: bool = True, strip_whitespace: bool = True) -> None:
        self.ignore_case = ignore_case
        self.strip_whitespace = strip_whitespace

    def _normalize(self, text: str) -> str:
        s = text
        if self.strip_whitespace:
            s = " ".join(s.split())
        if self.ignore_case:
            s = s.lower()
        return s

    def compute(self, prediction: str, reference: str | list[str], **kwargs: Any) -> float:
        norm_pred = self._normalize(prediction)
        refs = [reference] if isinstance(reference, str) else reference

        for ref in refs:
            if norm_pred == self._normalize(ref):
                return 1.0
        return 0.0

    def compute_details(
        self, prediction: str, reference: str | list[str], **kwargs: Any
    ) -> MetricResult:
        score = self.compute(prediction, reference, **kwargs)
        return MetricResult(
            metric_name=self.name,
            score=score,
            details={"matched": bool(score == 1.0)},
        )
