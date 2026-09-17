"""UrduMMLU external benchmark adapter."""

from __future__ import annotations

from collections.abc import Iterator

from urdu_eval.benchmarks.base import Benchmark
from urdu_eval.enums import Language, Script, TaskType
from urdu_eval.models import BenchmarkMetadata, Sample

URDU_MMLU_DOMAINS = {
    "stem": [
        "physics",
        "chemistry",
        "biology",
        "mathematics",
        "computer_science",
        "electrical_engineering",
        "college_physics",
        "college_chemistry",
        "astronomy",
    ],
    "humanities": [
        "history",
        "philosophy",
        "islamic_studies",
        "jurisprudence",
        "world_religions",
        "international_law",
        "urdu_literature",
        "pakistan_studies",
    ],
    "social_sciences": [
        "economics",
        "sociology",
        "political_science",
        "psychology",
        "geography",
        "public_relations",
        "human_sexuality",
    ],
    "profession": [
        "accounting",
        "management",
        "marketing",
        "clinical_knowledge",
        "medical_genetics",
        "professional_law",
        "professional_psychology",
    ],
    "other": ["general_knowledge", "everyday_facts", "logical_reasoning", "general_science"],
}


def resolve_domain_for_subject(subject: str) -> str:
    """Map a subject name to one of the 5 standard UrduMMLU macro-domains."""
    sub_lower = subject.lower().replace(" ", "_")
    for domain, subjects in URDU_MMLU_DOMAINS.items():
        if any(s in sub_lower or sub_lower in s for s in subjects):
            return domain.upper()
    return "OTHER"


class UrduMMLUAdapter(Benchmark):
    """Adapter for the public UrduMMLU benchmark from MBZUAI / community.

    Citation:
    UrduMMLU: Massive Multitask Language Understanding in Urdu (26,431 native questions across 5 domains).
    License: Creative Commons Attribution-ShareAlike 4.0 International.
    """

    def __init__(
        self,
        hf_dataset_name: str = "UrduMMLU/UrduMMLU",
        split: str = "test",
        subset: str | None = None,
        domain: str | None = None,
    ) -> None:
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.subset = subset
        self.domain = domain.lower() if domain else None

        domain_desc = f" [Domain: {domain.upper()}]" if domain else ""
        self.metadata = BenchmarkMetadata(
            id="urdummlu" if not domain else f"urdummlu-{domain.lower()}",
            name=f"UrduMMLU (External Dataset){domain_desc}",
            description=(
                "Massive Multitask Language Understanding benchmark in Urdu across 5 domains "
                "(STEM, Humanities, Social Sciences, Profession, Other)."
            ),
            languages=[Language.URDU],
            scripts=[Script.URDU],
            tasks=[TaskType.MMLU],
            metrics=["exact_match", "accuracy"],
            source=f"HuggingFace Datasets: {self.hf_dataset_name}",
            license="CC-BY-SA-4.0",
            version="1.0.0",
            provenance="https://huggingface.co/datasets/UrduMMLU/UrduMMLU",
            citation="UrduMMLU: Massive Multitask Language Understanding in Urdu (26,431 questions across 5 domains).",
            is_development_sample=False,
            dataset_scope="official",
            official_benchmark_size=26431,
        )

    def load_samples(self, max_samples: int | None = None) -> Iterator[Sample]:
        """Stream samples from HuggingFace datasets with domain annotations."""
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
            subject = item.get("subject", "")
            item_domain = resolve_domain_for_subject(subject)

            # Filter by domain if specified
            if self.domain and item_domain.lower() != self.domain:
                continue

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
                metadata={
                    "subject": subject,
                    "domain": item_domain,
                },
            )
            yield sample
            count += 1
            if max_samples is not None and count >= max_samples:
                break

    def count_samples(self) -> int:
        """Total verified questions in published UrduMMLU dataset."""
        return 26431
