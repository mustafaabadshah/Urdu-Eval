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


class UrBLiMPAdapter(Benchmark):
    """Adapter for UrBLiMP: Urdu Benchmark of Linguistic Minimal Pairs.

    5,696 minimal pairs covering 10 syntactic and morphosyntactic phenomena in Urdu,
    with reported 96.1% human inter-annotator agreement.

    Reference:
    UrBLiMP: A Linguistic Benchmark for Assessing Urdu Language Models.
    Task: Minimal Pair Discrimination (Accuracy of grammatical vs ungrammatical preference).
    """

    def __init__(
        self,
        phenomenon: str | None = None,
        hf_dataset_name: str = "urdu-nlp/urblimp",
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
                "phenomena (5,696 minimal pairs, 96.1% human agreement)."
            ),
            languages=[Language.URDU],
            scripts=[Script.URDU],
            tasks=[TaskType.MINIMAL_PAIR],
            metrics=["exact_match", "accuracy"],
            source=f"UrBLiMP Corpus ({self.hf_dataset_name})",
            license="CC-BY-4.0",
            version="1.0.0",
            provenance="https://huggingface.co/datasets/urdu-nlp/urblimp",
            is_development_sample=False,
        )

    def load_samples(self, max_samples: int | None = None) -> Iterator[Sample]:
        """Stream samples from HuggingFace dataset or bundled development set."""
        hf_available = False
        try:
            from datasets import load_dataset as hf_load_dataset

            ds = hf_load_dataset(self.hf_dataset_name, split=self.split, streaming=True)
            hf_available = True
        except Exception:
            hf_available = False

        if not hf_available:
            # Fall back gracefully to bundled development sample
            if self.use_dev_fallback and _DEV_SAMPLE_PATH.exists():
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
            raise ImportError(
                "The 'datasets' package is required to load UrBLiMP from HuggingFace. "
                "Install it with: pip install datasets"
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
        """Total verified minimal pairs in published UrBLiMP dataset."""
        return 5696
