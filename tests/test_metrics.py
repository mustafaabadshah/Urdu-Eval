"""Tests for evaluation metrics."""

import pytest

from urdu_eval.metrics import (
    get_metric,
    list_metrics,
    parse_judge_output,
)
from urdu_eval.metrics.judge import LLMJudgeMetric
from urdu_eval.providers.mock import MockProvider


def test_metric_registry() -> None:
    """Verify built-in metric discovery in registry."""
    metrics = list_metrics()
    assert "exact_match" in metrics
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "squad_f1" in metrics
    assert "bleu" in metrics
    assert "chrf" in metrics
    assert "rouge-1" in metrics
    assert "rouge-2" in metrics
    assert "rouge-l" in metrics
    assert "semantic_similarity" in metrics
    assert "llm_judge" in metrics


def test_exact_match_metric() -> None:
    """Test exact match scoring for Urdu and Roman Urdu."""
    em = get_metric("exact_match")

    # Urdu match with extra whitespace
    assert em.compute("  اسلام آباد  ", "اسلام آباد") == 1.0
    assert em.compute("لاہور", "اسلام آباد") == 0.0

    # Roman Urdu case insensitivity
    assert em.compute("islamabad", "Islamabad") == 1.0

    # Multiple references
    assert em.compute("Islam Abad", ["Islamabad", "Islam Abad"]) == 1.0


def test_f1_metric() -> None:
    """Test token F1 precision/recall calculation."""
    f1 = get_metric("f1")

    # Exact token match
    assert f1.compute("پاکستان ایک خوبصورت ملک ہے", "پاکستان ایک خوبصورت ملک ہے") == 1.0

    # Partial token overlap
    score = f1.compute("پاکستان ایک ملک ہے", "پاکستان ایک خوبصورت ملک ہے")
    assert 0.5 < score < 1.0

    # Disjoint tokens
    assert f1.compute("لاہور شہر", "اسلام آباد دارالحکومت") == 0.0


def test_bleu_metric() -> None:
    """Test BLEU translation score."""
    bleu = get_metric("bleu")

    pred = "Knowledge is an obligation upon every Muslim"
    ref = "Knowledge is an obligation upon every Muslim"
    assert bleu.compute(pred, ref) == pytest.approx(1.0, rel=1e-2)

    # Empty prediction
    assert bleu.compute("", ref) == 0.0


def test_chrf_metric() -> None:
    """Test chrF++ character and word n-gram metric."""
    chrf = get_metric("chrf")

    # High score on shared morphological roots in Roman Urdu
    score = chrf.compute("shukria", "shukriya")
    assert score > 0.45

    # Perfect match
    assert chrf.compute("اسلام آباد", "اسلام آباد") == pytest.approx(1.0, rel=1e-2)


def test_rouge_metrics() -> None:
    """Test ROUGE-1, ROUGE-2, and ROUGE-L."""
    r1 = get_metric("rouge-1")
    r2 = get_metric("rouge-2")
    rl = get_metric("rouge-l")

    pred = "علی سکول گیا اور سبق پڑھا"
    ref = "علی سکول گیا اور کتاب پڑھی"

    assert r1.compute(pred, ref) > 0.6
    assert r2.compute(pred, ref) > 0.4
    assert rl.compute(pred, ref) > 0.6


def test_semantic_similarity_metric() -> None:
    """Test semantic / character-word sequence similarity."""
    sem = get_metric("semantic_similarity")
    assert sem.compute("اسلام آباد", "اسلام آباد") == 1.0
    assert sem.compute("islamabad", "islamabad") == 1.0
    assert sem.compute("abc", "xyz") == 0.0


def test_judge_parser_clean_json() -> None:
    """Test parsing clean JSON judge output."""
    raw = '{"correctness": 4, "relevance": 2, "language_quality": 2, "instruction_following": 2, "total": 10, "explanation": "Perfect"}'
    res = parse_judge_output(raw)
    assert res.correctness == 4
    assert res.total == 10
    assert res.explanation == "Perfect"


def test_judge_parser_markdown_codeblock() -> None:
    """Test extracting JSON from markdown ```json ``` codeblock."""
    raw = """Here is my evaluation:
```json
{
  "correctness": 3,
  "relevance": 2,
  "language_quality": 1,
  "instruction_following": 2,
  "total": 8,
  "explanation": "Minor grammar error."
}
```
Thank you!"""
    res = parse_judge_output(raw)
    assert res.correctness == 3
    assert res.language_quality == 1
    assert res.total == 8


def test_judge_parser_clamping_and_fallback() -> None:
    """Test clamping out-of-bound scores and handling malformed strings."""
    # Score 99 clamped to max
    raw = '{"correctness": 99, "relevance": 2, "language_quality": 2, "instruction_following": 2, "total": 105}'
    res = parse_judge_output(raw)
    assert res.correctness == 4
    assert res.total == 10

    # Total gibberish fallback
    bad = "I think the model did okay, maybe 7/10."
    res_bad = parse_judge_output(bad)
    assert res_bad.total == 0
    assert "Failed to parse" in res_bad.explanation


def test_llm_judge_metric_with_mock_provider() -> None:
    """Test LLMJudgeMetric using MockProvider as evaluator."""
    judge_json = '{"correctness": 4, "relevance": 2, "language_quality": 2, "instruction_following": 2, "total": 10, "explanation": "All good"}'
    mock_judge = MockProvider(default_response=judge_json, canned_responses={})
    metric = LLMJudgeMetric(judge_provider=mock_judge)

    score = metric.compute("اسلام آباد", "اسلام آباد", prompt="دارالحکومت؟")
    assert score == 1.0  # 10 / 10 = 1.0

    details = metric.compute_details("اسلام آباد", "اسلام آباد", prompt="دارالحکومت؟")
    assert details.score == 1.0
    assert details.details["total_10"] == 10
    assert details.details["correctness"] == 4
