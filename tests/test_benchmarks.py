"""Tests for benchmark registry and custom benchmarks."""

from pathlib import Path

import pytest

from urdu_eval.benchmarks import (
    CustomBenchmark,
    get_benchmark,
    list_benchmarks,
)
from urdu_eval.enums import TaskType


def test_builtin_benchmark_discovery() -> None:
    """Test listing built-in benchmarks."""
    benchmarks = list_benchmarks()
    ids = [b.id for b in benchmarks]

    assert "urdu-qa" in ids
    assert "urdu-reasoning" in ids
    assert "urdu-translation" in ids
    assert "urdu-summary" in ids
    assert "urdu-roman" in ids
    assert "urdu-mmlu" in ids

    # All built-in dev suites must be marked as development sample
    for b in benchmarks:
        assert b.is_development_sample is True
        assert b.version == "0.1.0"


def test_get_benchmark_samples() -> None:
    """Test retrieving a benchmark and streaming its samples."""
    bm = get_benchmark("urdu-qa")
    assert bm.metadata.id == "urdu-qa"
    assert bm.metadata.tasks == [TaskType.QA]
    assert bm.count_samples() == 10

    samples = list(bm.load_samples())
    assert len(samples) == 10
    assert samples[0].id == "uqa_001"
    assert "پاکستان" in samples[0].prompt


def test_get_unknown_benchmark() -> None:
    """Verify error on unknown benchmark ID."""
    with pytest.raises(ValueError, match="was not found"):
        get_benchmark("completely-unknown-benchmark")


def test_custom_benchmark() -> None:
    """Test loading a user-provided custom benchmark JSONL."""
    sample_path = Path("examples/sample_qa.jsonl")
    custom_bm = CustomBenchmark(sample_path)

    assert custom_bm.metadata.id == "custom-sample_qa"
    assert custom_bm.metadata.is_development_sample is False
    assert custom_bm.count_samples() == 5

    samples = list(custom_bm.load_samples())
    assert len(samples) == 5
    assert samples[0].id == "pk_001"


def test_custom_benchmark_invalid_file() -> None:
    """Verify error when initializing CustomBenchmark with invalid file."""
    invalid_path = Path("examples/invalid_sample.jsonl")
    with pytest.raises(ValueError, match="is invalid"):
        CustomBenchmark(invalid_path)
