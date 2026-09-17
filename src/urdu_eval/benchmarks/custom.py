"""Custom user dataset benchmark implementation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from urdu_eval.benchmarks.base import Benchmark
from urdu_eval.dataset import compute_dataset_hash, load_dataset, validate_dataset
from urdu_eval.models import BenchmarkMetadata, Sample


class CustomBenchmark(Benchmark):
    """Benchmark created from a user-supplied JSONL file."""

    def __init__(self, file_path: str | Path, name: str | None = None) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Custom dataset file not found: {self.file_path}")

        val_result = validate_dataset(self.file_path)
        if not val_result.is_valid:
            raise ValueError(
                f"Custom dataset at '{self.file_path}' is invalid:\n{val_result.summary()}"
            )

        self._sample_count = val_result.total_samples
        dataset_hash = compute_dataset_hash(self.file_path)

        self.metadata = BenchmarkMetadata(
            id=f"custom-{self.file_path.stem}",
            name=name or f"Custom Benchmark ({self.file_path.name})",
            description=f"User-provided custom dataset from {self.file_path}",
            metrics=["exact_match", "f1"],
            source="Local File",
            license="User Defined",
            version="1.0",
            provenance=f"SHA-256:{dataset_hash}",
            dataset_sha256=dataset_hash,
            dataset_scope="official",
            is_development_sample=False,
        )

    def load_samples(self, max_samples: int | None = None) -> Iterator[Sample]:
        """Stream samples from the custom JSONL file."""
        return load_dataset(self.file_path, max_samples=max_samples)

    def count_samples(self) -> int:
        """Return total number of validated samples."""
        return self._sample_count
