"""Tests for UrBLiMP linguistic minimal pairs adapter."""

from urdu_eval.benchmarks.adapters.urblimp import UrBLiMPAdapter
from urdu_eval.benchmarks.registry import get_benchmark


def test_urblimp_adapter_initialization() -> None:
    """Verify UrBLiMP adapter metadata and sample count."""
    adapter = UrBLiMPAdapter()
    assert adapter.metadata.id == "urblimp"
    assert adapter.metadata.official_benchmark_size == 5696
    assert adapter.metadata.dataset_scope == "development"
    assert adapter.count_samples() == 10
    assert "UrBLiMP" in adapter.metadata.name
    assert "exact_match" in adapter.metadata.metrics
    assert "Adeeba" in adapter.metadata.citation


def test_urblimp_adapter_phenomena() -> None:
    """Verify UrBLiMP adapter supports phenomenon-specific instances."""
    adapter = UrBLiMPAdapter(phenomenon="subject_verb_agreement")
    assert adapter.metadata.id == "urblimp-subject-verb-agreement"
    assert "subject_verb_agreement" in adapter.metadata.name


def test_urblimp_load_dev_samples() -> None:
    """Verify UrBLiMP streams samples from bundled dev set when offline."""
    adapter = UrBLiMPAdapter()
    samples = list(adapter.load_samples(max_samples=5))
    assert len(samples) == 5
    assert samples[0].task.value == "minimal_pair"
    assert samples[0].reference == "A"
    assert "phenomenon" in samples[0].metadata


def test_urblimp_registry_lookup() -> None:
    """Verify UrBLiMP is retrievable via benchmark registry."""
    bm = get_benchmark("urblimp")
    assert bm.metadata.id == "urblimp"

    bm_phen = get_benchmark("urblimp-case-marking")
    assert bm_phen.metadata.id == "urblimp-case-marking"
