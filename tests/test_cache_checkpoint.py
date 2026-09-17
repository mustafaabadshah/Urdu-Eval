"""Tests for caching, retries, and checkpoint resumption."""

import tempfile
from pathlib import Path

from urdu_eval.enums import FailureCategory
from urdu_eval.models import ModelResponse, SampleResult
from urdu_eval.runner.cache import EvaluationCache, compute_cache_key
from urdu_eval.runner.checkpoint import CheckpointManager
from urdu_eval.runner.retry import execute_with_retry


def test_cache_key_computation() -> None:
    """Test deterministic cache key generation."""
    k1 = compute_cache_key("openai", "gpt-4o", "سوال", temperature=0.0)
    k2 = compute_cache_key("openai", "gpt-4o", "سوال", temperature=0.0)
    k3 = compute_cache_key("openai", "gpt-4o", "سوال", temperature=0.7)
    k4 = compute_cache_key("mock", "mock-model", "سوال", temperature=0.0)

    assert k1 == k2
    assert k1 != k3
    assert k1 != k4


def test_evaluation_cache_crud() -> None:
    """Test setting, getting, and clearing cached responses in SQLite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EvaluationCache(cache_dir=tmpdir)
        key = "test_key_123"

        # Initially empty
        assert cache.get(key) is None

        resp = ModelResponse(
            text="جواب",
            model="test-m",
            provider="mock",
            latency_ms=10.0,
        )
        cache.set(key, resp)

        cached = cache.get(key)
        assert cached is not None
        assert cached.text == "جواب"
        assert cached.model == "test-m"

        cache.clear()
        assert cache.get(key) is None


def test_checkpoint_manager_resume() -> None:
    """Test checkpoint recording and resumption across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        mgr1 = CheckpointManager(run_dir)
        assert mgr1.completed_count == 0

        res1 = SampleResult(
            sample_id="s_01",
            prompt="سوال 1",
            reference="جواب 1",
            prediction="جواب 1",
            metrics={"exact_match": 1.0},
            failure_category=FailureCategory.CORRECT,
        )
        mgr1.record_sample(res1)
        assert mgr1.is_completed("s_01") is True
        assert mgr1.is_completed("s_02") is False
        assert mgr1.completed_count == 1

        # Simulate interruption & resume by creating new manager pointing to same run_dir
        mgr2 = CheckpointManager(run_dir)
        assert mgr2.completed_count == 1
        assert mgr2.is_completed("s_01") is True

        res2 = SampleResult(
            sample_id="s_02",
            prompt="سوال 2",
            reference="جواب 2",
            prediction="جواب 2",
            metrics={"exact_match": 1.0},
        )
        mgr2.record_sample(res2)
        assert mgr2.completed_count == 2
        results = mgr2.get_completed_results()
        assert len(results) == 2
        assert results[0].sample_id == "s_01"
        assert results[1].sample_id == "s_02"


def test_retry_with_backoff_success_after_failure() -> None:
    """Test retry handler successfully completes after initial transient error."""
    attempts = 0

    def flaky_func() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("HTTP 429: Too Many Requests")
        return "success"

    res = execute_with_retry(flaky_func, max_retries=3, initial_delay=0.01)
    assert res == "success"
    assert attempts == 2
