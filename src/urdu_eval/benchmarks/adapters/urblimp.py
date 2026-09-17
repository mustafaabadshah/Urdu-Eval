"""UrBLiMP: Urdu Benchmark of Linguistic Minimal Pairs adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from urdu_eval.benchmarks.base import Benchmark
from urdu_eval.dataset import load_dataset
from urdu_eval.enums import Language, Script, TaskType
from urdu_eval.models import BenchmarkMetadata, Sample

URBLIMP_PHENOMENA = [
    "subject_verb_agreement",
    "case_marking",
    "word_order",
    "pro_drop",
    "verb_subcategorization",
    "anaphora_binding",
    "coordination",
    "filler_gap_dependency",
    "negation_scope",
    "tense_aspect_concord",
]

_DEV_SAMPLE_PATH = Path(__file__).parent.parent / "dev_samples" / "urblimp.jsonl"


CITATION = (
    "Adeeba, F., Dillon, B., Sajjad, H., & Bhatt, R. (2026). "
    "UrBLiMP: A Benchmark for Evaluating the Linguistic Competence of Large Language Models in Urdu. "
    "Findings of the Association for Computational Linguistics: ACL 2026 (arXiv:2508.01006)."
)


class UrBLiMPAdapter(Benchmark):
    """Adapter for UrBLiMP: Urdu Benchmark of Linguistic Minimal Pairs.

    5,696 minimal pairs covering 10 syntactic and morphosyntactic phenomena in Urdu,
    with reported 96.1% human inter-annotator agreement.

    Reference:
        Adeeba, F., Dillon, B., Sajjad, H., & Bhatt, R. (2026).
        UrBLiMP: A Benchmark for Evaluating the Linguistic Competence of Large Language Models in Urdu.
        Findings of the Association for Computational Linguistics: ACL 2026 (arXiv:2508.01006).

    Task: Minimal Pair Discrimination (Accuracy of grammatical vs ungrammatical preference).
    """

    OFFICIAL_BENCHMARK_SIZE: int = 5696

    def __init__(
        self,
        phenomenon: str | None = None,
        hf_dataset_name: str | None = None,
        split: str = "test",
        use_dev_fallback: bool = True,
    ) -> None:
        self.phenomenon = phenomenon.lower().replace("-", "_") if phenomenon else None
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.use_dev_fallback = use_dev_fallback

        phen_desc = f" [Phenomenon: {self.phenomenon}]" if self.phenomenon else ""
        self.metadata = BenchmarkMetadata(
            id="urblimp" if not self.phenomenon else f"urblimp-{self.phenomenon.replace('_', '-')}",
            name=f"UrBLiMP (Linguistic Minimal Pairs){phen_desc}",
            description=(
                "Urdu Benchmark of Linguistic Minimal Pairs testing 10 syntactic and morphosyntactic "
                "phenomena (5,696 minimal pairs, 96.1% human agreement; Adeeba et al., ACL 2026)."
            ),
            languages=[Language.URDU],
            scripts=[Script.URDU],
            tasks=[TaskType.MINIMAL_PAIR],
            metrics=["exact_match", "accuracy"],
            source="UrBLiMP (Adeeba et al., ACL 2026, arXiv:2508.01006)",
            license="CC-BY-4.0",
            version="1.0.0",
            provenance="Adeeba et al. (ACL 2026) / arXiv:2508.01006",
            citation=CITATION,
            is_development_sample=True,
            dataset_scope="development",
            official_benchmark_size=self.OFFICIAL_BENCHMARK_SIZE,
        )

    def load_samples(self, max_samples: int | None = None) -> Iterator[Sample]:
        """Stream samples from configured dataset or bundled development set."""
        hf_available = False
        if self.hf_dataset_name:
            try:
                from datasets import load_dataset as hf_load_dataset

                ds = hf_load_dataset(self.hf_dataset_name, split=self.split, streaming=True)
                hf_available = True
                self.metadata.is_development_sample = False
                self.metadata.dataset_scope = "official"
            except Exception:
                hf_available = False

        if not hf_available:
            # Fall back to bundled development sample with loud explicit warning
            if self.use_dev_fallback and _DEV_SAMPLE_PATH.exists():
                import sys

                sys.stderr.write(
                    "\n⚠️  UrBLiMP NOTICE: Full dataset unavailable. Using bundled DEVELOPMENT samples (N=10).\n"
                    "   This run is NOT an official UrBLiMP evaluation.\n"
                    "   To evaluate all 5,696 pairs, provide the official dataset (Adeeba et al., ACL 2026) via: --dataset <path_to_urblimp.jsonl>\n\n"
                )
                self.metadata.is_development_sample = True
                self.metadata.dataset_scope = "development"
                count = 0
                for s in load_dataset(_DEV_SAMPLE_PATH):
                    item_phen = s.metadata.get("phenomenon", "").lower().replace("-", "_")
                    if self.phenomenon and item_phen != self.phenomenon:
                        continue
                    yield s
                    count += 1
                    if max_samples is not None and count >= max_samples:
                        break
                return
            raise FileNotFoundError(
                "UrBLiMP dataset file not found. Provide the official dataset via --dataset <path>."
            )

        count = 0
        for item in ds:
            item_phen = item.get("phenomenon", "").lower().replace("-", "_")
            if self.phenomenon and item_phen != self.phenomenon:
                continue

            sen_good = item.get("sentence_good", "")
            sen_bad = item.get("sentence_bad", "")

            # Deterministic presentation order
            prompt_text = (
                f"درج ذیل میں سے کون سا جملہ گرامر کے لحاظ سے درست ہے؟\n"
                f"A) {sen_good}\n"
                f"B) {sen_bad}\n"
                f"صرف درست آپشن کا حرف (A یا B) لکھیں۔"
            )

            sample = Sample(
                id=str(item.get("pair_id", f"urblimp_{count}")),
                task=TaskType.MINIMAL_PAIR,
                language=Language.URDU,
                script=Script.URDU,
                prompt=prompt_text,
                reference="A",
                options=[sen_good, sen_bad],
                metadata={
                    "phenomenon": item_phen,
                    "sub_phenomenon": item.get("sub_phenomenon", ""),
                    "pair_id": item.get("pair_id", ""),
                },
            )
            yield sample
            count += 1
            if max_samples is not None and count >= max_samples:
                break

    def count_samples(self) -> int:
        """Sample count for currently active dataset scope."""
        if self.metadata.is_development_sample:
            return 10
        return self.OFFICIAL_BENCHMARK_SIZE
