"""Base metric protocol and registry."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from urdu_eval.models import MetricResult


@runtime_checkable
class Metric(Protocol):
    """Protocol that all evaluation metrics must implement."""

    name: str

    def compute(self, prediction: str, reference: str | list[str], **kwargs: Any) -> float:
        """Compute the scalar metric score for a prediction against reference(s)."""
        ...

    def compute_details(
        self, prediction: str, reference: str | list[str], **kwargs: Any
    ) -> MetricResult:
        """Compute metric with detailed diagnostic breakdown."""
        ...


_METRIC_REGISTRY: dict[str, type[Metric]] = {}


def register_metric(name: str):
    """Decorator to register a metric class."""

    def decorator(cls: type[Metric]):
        _METRIC_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_metric(name: str, **kwargs: Any) -> Metric:
    """Retrieve an instantiated metric by name."""
    metric_key = name.lower()
    if metric_key not in _METRIC_REGISTRY:
        _discover_builtin_metrics()

    if metric_key not in _METRIC_REGISTRY:
        available = ", ".join(sorted(_METRIC_REGISTRY.keys()))
        raise ValueError(f"Metric '{name}' is not registered. Available metrics: {available}")

    metric_cls = _METRIC_REGISTRY[metric_key]
    return metric_cls(**kwargs)


def list_metrics() -> list[str]:
    """List all registered metric names."""
    _discover_builtin_metrics()
    return sorted(_METRIC_REGISTRY.keys())


def _discover_builtin_metrics() -> None:
    """Import built-in metric modules to register them."""
    try:
        from urdu_eval.metrics.exact_match import ExactMatchMetric  # noqa: F401
    except ImportError:
        pass

    try:
        from urdu_eval.metrics.f1 import F1Metric, SquadF1Metric  # noqa: F401
    except ImportError:
        pass

    try:
        from urdu_eval.metrics.bleu import BLEUMetric  # noqa: F401
    except ImportError:
        pass

    try:
        from urdu_eval.metrics.chrf import ChrFMetric  # noqa: F401
    except ImportError:
        pass

    try:
        from urdu_eval.metrics.rouge import ROUGE1Metric, ROUGE2Metric, ROUGELMetric  # noqa: F401
    except ImportError:
        pass

    try:
        from urdu_eval.metrics.semantic import SemanticSimilarityMetric  # noqa: F401
    except ImportError:
        pass

    try:
        from urdu_eval.metrics.judge import LLMJudgeMetric  # noqa: F401
    except ImportError:
        pass
