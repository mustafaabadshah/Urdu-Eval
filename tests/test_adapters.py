"""Tests for external benchmark adapters."""

import pytest

from urdu_eval.benchmarks.adapters import UrduBenchAdapter, UrduMMLUAdapter
from urdu_eval.enums import TaskType


def test_urdu_mmlu_adapter_metadata() -> None:
    """Verify UrduMMLUAdapter exposes proper metadata, citations, and provenance."""
    adapter = UrduMMLUAdapter()
    assert adapter.metadata.id in ("urdummlu", "urdu-mmlu-external")
    assert adapter.metadata.tasks == [TaskType.MMLU]
    assert adapter.metadata.license == "CC-BY-SA-4.0"
    assert "https://huggingface.co/datasets/UrduMMLU" in adapter.metadata.provenance
    assert adapter.metadata.is_development_sample is False

    # Loading without 'datasets' library installed raises clean informative error
    try:
        import datasets  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="datasets"):
            list(adapter.load_samples(max_samples=1))


def test_urdu_bench_adapter_metadata() -> None:
    """Verify UrduBenchAdapter exposes proper metadata, tasks, and provenance."""
    adapter = UrduBenchAdapter()
    assert adapter.metadata.id == "urdu-bench-reasoning"
    assert adapter.metadata.tasks == [TaskType.REASONING]
    assert adapter.metadata.is_development_sample is False
    assert adapter.count_samples() == 1000
