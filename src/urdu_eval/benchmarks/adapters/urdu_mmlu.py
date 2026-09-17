"""UrduMMLU external benchmark adapter."""

from __future__ import annotations

from collections.abc import Iterator

from urdu_eval.benchmarks.base import Benchmark
from urdu_eval.enums import Language, Script, TaskType
from urdu_eval.models import BenchmarkMetadata, Sample


class UrduMMLUAdapter(Benchmark):
    """Adapter for the public UrduMMLU benchmark.

    Citation:
    UrduMMLU: Massive Multitask Language Understanding in Urdu.
    License: Creative Commons Attribution-ShareAlike 4.0 International.
    """

    def __init__(
        self,
        hf_dataset_name: str = "UrduMMLU/UrduMMLU",
        split: str = "test",
        subset: str | None = None,
    ) -> None:
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.subset = subset

        self.metadata = BenchmarkMetadata(
            id="urdu-mmlu-external",
            name="UrduMMLU (External Adapter)",
            description=(
                "Massive Multitask Language Understanding benchmark translated and localized "
                "into Urdu across 57 subjects."
            ),
            languages=[Language.URDU],
            scripts=[Script.URDU],
            tasks=[TaskType.MMLU],
            metrics=["exact_match", "accuracy"],
            source=f"HuggingFace Datasets: {self.hf_dataset_name}",
            license="CC-BY-SA-4.0",
            version="1.0",
            provenance="https://huggingface.co/datasets/UrduMMLU/UrduMMLU",
            is_development_sample=False,
        )

    def load_samples(self, max_samples: int | None = None) -> Iterator[Sample]:
        """Stream samples from HuggingFace datasets."""
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to load UrduMMLU from HuggingFace. "
                "Install it with: pip install datasets"
            ) from exc

        ds = hf_load_dataset(self.hf_dataset_name, self.subset, split=self.split, streaming=True)
        count = 0
        for item in ds:
            options = [item.get("A", ""), item.get("B", ""), item.get("C", ""), item.get("D", "")]
            prompt_text = (
                f"{item.get('question', '')}\n"
                f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n"
                f"صرف درست آپشن کا حرف لکھیں۔"
            )
            sample = Sample(
                id=str(item.get("id", f"ummlu_{count}")),
                task=TaskType.MMLU,
                language=Language.URDU,
                script=Script.URDU,
                prompt=prompt_text,
                reference=str(item.get("answer", "")).strip(),
                options=options,
                metadata={"subject": item.get("subject", "")},
            )
            yield sample
            count += 1
            if max_samples is not None and count >= max_samples:
                break

    def count_samples(self) -> int:
        """Estimate total samples."""
        return 14000
