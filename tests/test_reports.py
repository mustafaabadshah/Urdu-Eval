"""Tests for Markdown, HTML, and JSON reports."""

from urdu_eval.enums import FailureCategory
from urdu_eval.models import (
    BenchmarkMetadata,
    ModelConfig,
    RunMetadata,
    RunResult,
    SampleResult,
)
from urdu_eval.reports import generate_html_report, generate_markdown_report


def _create_mock_run_result() -> RunResult:
    b_meta = BenchmarkMetadata(
        id="urdu-qa",
        name="Urdu QA",
        description="Urdu Question Answering",
        source="Curated",
        license="Apache-2.0",
        version="0.1.0",
    )
    m_cfg = ModelConfig(provider="mock", model="mock-urdu-model")
    r_meta = RunMetadata(
        run_id="run_test_report_123",
        timestamp="2026-09-17T12:00:00Z",
        urdu_eval_version="0.1.0",
        benchmark=b_meta,
        dataset_hash="hash_123",
        model_config=m_cfg,
        python_version="3.12.3",
        platform="Windows",
    )
    s1 = SampleResult(
        sample_id="s1",
        prompt="پاکستان کا دارالحکومت کیا ہے؟",
        reference="اسلام آباد",
        prediction="اسلام آباد",
        metrics={"exact_match": 1.0, "f1": 1.0},
        latency_ms=15.0,
        failure_category=FailureCategory.CORRECT,
    )
    s2 = SampleResult(
        sample_id="s2",
        prompt="لاہور کس ملک میں ہے؟",
        reference="پاکستان",
        prediction="بھارت",
        metrics={"exact_match": 0.0, "f1": 0.0},
        latency_ms=20.0,
        failure_category=FailureCategory.INCORRECT,
    )
    return RunResult(
        run_id="run_test_report_123",
        metadata=r_meta,
        metrics={"exact_match": 0.5, "f1": 0.5},
        samples=[s1, s2],
        failure_summary={"correct": 1, "incorrect": 1},
        total_samples=2,
        failed_samples=0,
        mean_latency_ms=17.5,
    )


def test_generate_markdown_report() -> None:
    """Test generating Markdown report from RunResult."""
    res = _create_mock_run_result()
    md = generate_markdown_report(res)

    assert "# UrduEval Evaluation Report" in md
    assert "mock-urdu-model" in md
    assert "exact_match" in md
    assert "اسلام آباد" in md
    assert "50.0%" in md


def test_generate_html_report() -> None:
    """Test generating standalone HTML report from RunResult."""
    res = _create_mock_run_result()
    html_doc = generate_html_report(res)

    assert "<!DOCTYPE html>" in html_doc
    assert "UrduEval Benchmark Report" in html_doc
    assert "mock-urdu-model" in html_doc
    assert "EXACT_MATCH" in html_doc
    assert "Sample-Level Inspector" in html_doc
    assert "s1" in html_doc
    assert "s2" in html_doc
