"""Tests for core Pydantic data models."""

import pytest
from pydantic import ValidationError

from urdu_eval.enums import (
    FailureCategory,
    Language,
    Script,
    TaskType,
    TranslationDirection,
)
from urdu_eval.models import (
    BenchmarkMetadata,
    HumanRating,
    JudgeResult,
    LeaderboardEntry,
    ModelConfig,
    ModelResponse,
    RunMetadata,
    RunResult,
    Sample,
    SampleResult,
)


def test_sample_creation_urdu() -> None:
    """Test creating a valid Urdu sample with defaults."""
    sample = Sample(
        id="pk_001",
        prompt="پاکستان کا دارالحکومت کیا ہے؟",
        reference="اسلام آباد",
    )
    assert sample.id == "pk_001"
    assert sample.task == TaskType.QA
    assert sample.language == Language.URDU
    assert sample.script == Script.URDU
    assert sample.reference == "اسلام آباد"
    assert sample.options is None


def test_sample_creation_roman_urdu() -> None:
    """Test creating a Roman Urdu sample with translation direction."""
    sample = Sample(
        id="ru_001",
        task=TaskType.TRANSLATION,
        language=Language.URDU,
        script=Script.ROMAN_URDU,
        prompt="Pakistan ka darul hukoomat kya hai?",
        reference=["Islamabad", "Islam Abad"],
        direction=TranslationDirection.URDU_TO_ENGLISH,
        metadata={"domain": "geography"},
    )
    assert sample.script == Script.ROMAN_URDU
    assert isinstance(sample.reference, list)
    assert len(sample.reference) == 2
    assert sample.metadata["domain"] == "geography"


def test_sample_missing_required_fields() -> None:
    """Verify that missing required prompt or reference raises ValidationError."""
    with pytest.raises(ValidationError):
        Sample.model_validate({"id": "err_01", "prompt": "سوال"})

    with pytest.raises(ValidationError):
        Sample.model_validate({"id": "err_02", "reference": "جواب"})


def test_model_config_and_response() -> None:
    """Test ModelConfig and ModelResponse initialization."""
    cfg = ModelConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.0,
        max_tokens=256,
    )
    assert cfg.provider == "openai"
    assert cfg.temperature == 0.0

    resp = ModelResponse(
        text="اسلام آباد",
        model="gpt-4o",
        provider="openai",
        latency_ms=450.5,
        input_tokens=20,
        output_tokens=5,
    )
    assert resp.text == "اسلام آباد"
    assert resp.latency_ms == 450.5
    assert resp.finish_reason == "stop"


def test_judge_result_validation() -> None:
    """Test JudgeResult field constraints."""
    valid_judge = JudgeResult(
        correctness=4,
        relevance=2,
        language_quality=2,
        instruction_following=2,
        total=10,
        explanation="Perfect answer in Urdu.",
        raw_response='{"correctness": 4, "relevance": 2, "language_quality": 2, "instruction_following": 2, "total": 10}',
    )
    assert valid_judge.total == 10

    # Test out-of-range correctness (> 4)
    with pytest.raises(ValidationError):
        JudgeResult(
            correctness=5,
            relevance=2,
            language_quality=2,
            instruction_following=2,
            total=11,
        )

    # Test negative relevance (< 0)
    with pytest.raises(ValidationError):
        JudgeResult(
            correctness=3,
            relevance=-1,
            language_quality=2,
            instruction_following=2,
            total=6,
        )


def test_benchmark_metadata() -> None:
    """Test BenchmarkMetadata creation."""
    meta = BenchmarkMetadata(
        id="urdu-qa",
        name="Urdu QA",
        description="Urdu Question Answering Benchmark",
        source="Curated",
        license="Apache-2.0",
        version="0.1.0",
        metrics=["exact_match", "f1"],
    )
    assert meta.id == "urdu-qa"
    assert meta.metrics == ["exact_match", "f1"]
    assert meta.languages == [Language.URDU]


def test_run_result_and_leaderboard_entry() -> None:
    """Test full RunResult assembly and serialization."""
    benchmark = BenchmarkMetadata(
        id="urdu-qa",
        name="Urdu QA",
        description="Urdu QA",
        source="Curated",
        license="Apache-2.0",
        version="0.1.0",
    )
    m_cfg = ModelConfig(provider="mock", model="mock-model")
    run_meta = RunMetadata(
        run_id="run_12345",
        timestamp="2026-09-17T12:00:00Z",
        urdu_eval_version="0.1.0",
        benchmark=benchmark,
        dataset_hash="a1b2c3d4",
        model_config=m_cfg,
        python_version="3.12.3",
        platform="Windows",
    )
    sample_res = SampleResult(
        sample_id="pk_001",
        prompt="پاکستان کا دارالحکومت کیا ہے؟",
        reference="اسلام آباد",
        prediction="اسلام آباد",
        metrics={"exact_match": 1.0, "f1": 1.0},
        latency_ms=12.0,
        failure_category=FailureCategory.CORRECT,
    )
    run_result = RunResult(
        run_id="run_12345",
        metadata=run_meta,
        metrics={"exact_match": 1.0, "f1": 1.0},
        samples=[sample_res],
        failure_summary={"correct": 1},
        total_samples=1,
        failed_samples=0,
        mean_latency_ms=12.0,
    )
    assert run_result.run_id == "run_12345"
    assert run_result.metrics["exact_match"] == 1.0

    entry = LeaderboardEntry(
        model="mock-model",
        provider="mock",
        benchmark_id="urdu-qa",
        benchmark_version="0.1.0",
        samples=1,
        metrics={"exact_match": 1.0},
        latency_ms=12.0,
        date="2026-09-17",
        run_id="run_12345",
    )
    assert entry.model == "mock-model"

    rating = HumanRating(
        sample_id="pk_001",
        annotator_id_hash="anon_hash_01",
        correctness=1.0,
        fluency=1.0,
        relevance=1.0,
    )
    assert rating.sample_id == "pk_001"
