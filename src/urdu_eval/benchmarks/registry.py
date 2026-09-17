"""Benchmark registry and built-in benchmark definitions."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from urdu_eval.benchmarks.base import Benchmark
from urdu_eval.dataset import compute_dataset_hash, load_dataset, validate_dataset
from urdu_eval.enums import Language, Script, TaskType
from urdu_eval.models import BenchmarkMetadata, Sample

_DEV_SAMPLES_DIR = Path(__file__).parent / "dev_samples"


class FileBenchmark(Benchmark):
    """A benchmark initialized from a known JSONL file with explicit metadata."""

    def __init__(self, metadata: BenchmarkMetadata, file_path: Path) -> None:
        self.metadata = metadata
        self.file_path = file_path
        if not self.file_path.exists():
            raise FileNotFoundError(f"Benchmark sample file not found: {self.file_path}")

        # Compute hash and count
        self.metadata.provenance = f"SHA-256:{compute_dataset_hash(self.file_path)}"
        val = validate_dataset(self.file_path)
        self._count = val.total_samples

    def load_samples(self, max_samples: int | None = None) -> Iterator[Sample]:
        return load_dataset(self.file_path, max_samples=max_samples)

    def count_samples(self) -> int:
        return self._count


_BENCHMARK_REGISTRY: dict[str, Benchmark] = {}


def register_benchmark(benchmark: Benchmark) -> None:
    """Register a Benchmark instance."""
    _BENCHMARK_REGISTRY[benchmark.metadata.id.lower()] = benchmark


def get_benchmark(benchmark_id: str) -> Benchmark:
    """Retrieve a benchmark by ID."""
    _ensure_builtin_benchmarks()
    bid = benchmark_id.lower()

    if bid in {"urdummlu", "urdu-mmlu-external"}:
        from urdu_eval.benchmarks.adapters.urdu_mmlu import UrduMMLUAdapter

        return UrduMMLUAdapter()

    if bid.startswith("urdummlu-"):
        from urdu_eval.benchmarks.adapters.urdu_mmlu import UrduMMLUAdapter

        domain = bid.split("-", 1)[1]
        return UrduMMLUAdapter(domain=domain)

    if bid not in _BENCHMARK_REGISTRY:
        available = ", ".join(
            sorted(_BENCHMARK_REGISTRY.keys())
            + ["urdummlu", "urdummlu-stem", "urdummlu-humanities"]
        )
        raise ValueError(
            f"Benchmark '{benchmark_id}' was not found. Available benchmarks:\n  {available}"
        )
    return _BENCHMARK_REGISTRY[bid]


def list_benchmarks() -> list[BenchmarkMetadata]:
    """List metadata for all registered benchmarks."""
    _ensure_builtin_benchmarks()
    return [b.metadata for b in _BENCHMARK_REGISTRY.values()]


def _ensure_builtin_benchmarks() -> None:
    """Register built-in development benchmark sample suites if not already registered."""
    if _BENCHMARK_REGISTRY:
        return

    # 1. urdu-qa
    qa_meta = BenchmarkMetadata(
        id="urdu-qa",
        name="Urdu QA (Dev Sample)",
        description="Urdu Question Answering development sample suite.",
        languages=[Language.URDU],
        scripts=[Script.URDU],
        tasks=[TaskType.QA],
        metrics=["exact_match", "f1"],
        source="Curated Pakistan Geography, History & General Knowledge",
        license="Apache-2.0",
        version="0.1.0",
        is_development_sample=True,
    )
    register_benchmark(FileBenchmark(qa_meta, _DEV_SAMPLES_DIR / "urdu_qa.jsonl"))

    # 2. urdu-reasoning
    rs_meta = BenchmarkMetadata(
        id="urdu-reasoning",
        name="Urdu Reasoning (Dev Sample)",
        description="Urdu multi-step arithmetic and logical reasoning development sample suite.",
        languages=[Language.URDU],
        scripts=[Script.URDU],
        tasks=[TaskType.REASONING],
        metrics=["exact_match", "f1"],
        source="Curated Mathematical & Logical Puzzles in Urdu",
        license="Apache-2.0",
        version="0.1.0",
        is_development_sample=True,
    )
    register_benchmark(FileBenchmark(rs_meta, _DEV_SAMPLES_DIR / "urdu_reasoning.jsonl"))

    # 3. urdu-translation
    tr_meta = BenchmarkMetadata(
        id="urdu-translation",
        name="Urdu Translation (Dev Sample)",
        description="Urdu ↔ English bidirectional translation sample suite.",
        languages=[Language.URDU, Language.ENGLISH],
        scripts=[Script.URDU, Script.LATIN],
        tasks=[TaskType.TRANSLATION],
        metrics=["bleu", "chrf++", "rouge-l"],
        source="Curated Sentences & Idiomatic Expressions",
        license="Apache-2.0",
        version="0.1.0",
        is_development_sample=True,
    )
    register_benchmark(FileBenchmark(tr_meta, _DEV_SAMPLES_DIR / "urdu_translation.jsonl"))

    # 4. urdu-summary
    sm_meta = BenchmarkMetadata(
        id="urdu-summary",
        name="Urdu Summarization (Dev Sample)",
        description="Urdu short-paragraph summarization sample suite.",
        languages=[Language.URDU],
        scripts=[Script.URDU],
        tasks=[TaskType.SUMMARIZATION],
        metrics=["rouge-1", "rouge-2", "rouge-l", "bleu"],
        source="Curated Informational Passages",
        license="Apache-2.0",
        version="0.1.0",
        is_development_sample=True,
    )
    register_benchmark(FileBenchmark(sm_meta, _DEV_SAMPLES_DIR / "urdu_summary.jsonl"))

    # 5. urdu-roman
    rm_meta = BenchmarkMetadata(
        id="urdu-roman",
        name="Roman Urdu QA (Dev Sample)",
        description="Roman Urdu transliterated QA and orthographic comprehension suite.",
        languages=[Language.URDU],
        scripts=[Script.ROMAN_URDU],
        tasks=[TaskType.QA],
        metrics=["exact_match", "f1", "chrf++"],
        source="Curated Roman Urdu Everyday QA",
        license="Apache-2.0",
        version="0.1.0",
        is_development_sample=True,
    )
    register_benchmark(FileBenchmark(rm_meta, _DEV_SAMPLES_DIR / "urdu_roman.jsonl"))

    # 6. urdu-mmlu
    mmlu_meta = BenchmarkMetadata(
        id="urdu-mmlu",
        name="Urdu MMLU Sample (Dev Sample)",
        description="Urdu multiple-choice domain knowledge suite.",
        languages=[Language.URDU],
        scripts=[Script.URDU],
        tasks=[TaskType.MMLU],
        metrics=["exact_match", "accuracy"],
        source="Curated Science & Humanities Multiple-Choice Sample",
        license="Apache-2.0",
        version="0.1.0",
        is_development_sample=True,
    )
    register_benchmark(FileBenchmark(mmlu_meta, _DEV_SAMPLES_DIR / "urdu_mmlu.jsonl"))
