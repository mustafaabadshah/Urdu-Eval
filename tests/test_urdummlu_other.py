"""Tests for UrduMMLU 5-domain adapter including Other domain."""

from urdu_eval.benchmarks.adapters.urdu_mmlu import (
    UrduMMLUAdapter,
    resolve_domain_for_subject,
)
from urdu_eval.benchmarks.registry import get_benchmark


def test_resolve_domain_all_5() -> None:
    """Verify all 5 UrduMMLU macro-domains resolve properly."""
    assert resolve_domain_for_subject("physics") == "STEM"
    assert resolve_domain_for_subject("islamic_studies") == "HUMANITIES"
    assert resolve_domain_for_subject("political_science") == "SOCIAL_SCIENCES"
    assert resolve_domain_for_subject("marketing") == "PROFESSION"
    assert resolve_domain_for_subject("general_knowledge") == "OTHER"
    assert resolve_domain_for_subject("everyday_facts") == "OTHER"


def test_urdummlu_other_adapter() -> None:
    """Verify UrduMMLU Other domain adapter initialization and registry lookup."""
    adapter = UrduMMLUAdapter(domain="other")
    assert adapter.metadata.id == "urdummlu-other"
    assert "OTHER" in adapter.metadata.name
    assert adapter.count_samples() == 26431

    bm = get_benchmark("urdummlu-other")
    assert bm.metadata.id == "urdummlu-other"
