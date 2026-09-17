"""UrduBench reasoning external benchmark adapter."""

from __future__ import annotations

from collections.abc import Iterator

from urdu_eval.benchmarks.base import Benchmark
from urdu_eval.enums import Language, Script, TaskType
from urdu_eval.models import BenchmarkMetadata, Sample


class UrduBenchAdapter(Benchmark):
    """Adapter for UrduBench reasoning benchmarks.

    Provenance: UrduBench project (urdubench.com).
    License: Permissive / Academic evaluation.
    """

    def __init__(
        self,
        hf_dataset_name: str = "urdu-bench/reasoning",
        split: str = "test",
    ) -> None:
        self.hf_dataset_name = hf_dataset_name
        self.split = split

        self.metadata = BenchmarkMetadata(
            id="urdu-bench-reasoning",
            name="UrduBench Reasoning (External Adapter)",
            description="Complex multi-step reasoning and logical deductions in Urdu.",
            languages=[Language.URDU],
            scripts=[Script.URDU],
            tasks=[TaskType.REASONING],
            metrics=["exact_match", "f1"],
            source=f"HuggingFace Datasets: {self.hf_dataset_name}",
            license="Research / Open Access",
            version="1.0",
            provenance="https://github.com/urdu-bench",
            is_development_sample=False,
        )

    def load_samples(self, max_samples: int | None = None) -> Iterator[Sample]:
        """Stream reasoning samples from dataset."""
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to load UrduBench. "
                "Install it with: pip install datasets"
            ) from exc

        ds = hf_load_dataset(self.hf_dataset_name, split=self.split, streaming=True)
        count = 0
        for item in ds:
            sample = Sample(
                id=str(item.get("id", f"ub_{count}")),
                task=TaskType.REASONING,
                language=Language.URDU,
                script=Script.URDU,
                prompt=item.get("question") or item.get("prompt", ""),
                reference=item.get("answer") or item.get("reference", ""),
                metadata={"subtask": item.get("category", "reasoning")},
            )
            yield sample
            count += 1
            if max_samples is not None and count >= max_samples:
                break

    def count_samples(self) -> int:
        return 1000
