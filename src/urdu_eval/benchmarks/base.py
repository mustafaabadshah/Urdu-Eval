"""Benchmark protocol and base classes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from urdu_eval.models import BenchmarkMetadata, Sample


@runtime_checkable
class Benchmark(Protocol):
    """Protocol for all evaluation benchmark suites."""

    metadata: BenchmarkMetadata

    def load_samples(self, max_samples: int | None = None) -> Iterator[Sample]:
        """Stream benchmark samples."""
        ...

    def count_samples(self) -> int:
        """Count total available samples."""
        ...
