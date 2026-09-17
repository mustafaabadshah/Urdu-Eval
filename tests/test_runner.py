"""Tests for EvaluationRunner execution and persistence."""

import json
import tempfile
from pathlib import Path

from urdu_eval.benchmarks import get_benchmark
from urdu_eval.models import ModelConfig, RunConfig
from urdu_eval.providers.mock import MockProvider
from urdu_eval.runner import EvaluationRunner


def test_runner_execution_with_mock_provider() -> None:
    """Test full evaluation pipeline on built-in urdu-qa benchmark with MockProvider."""
    with tempfile.TemporaryDirectory() as tmpdir:
        benchmark = get_benchmark("urdu-qa")
        model_cfg = ModelConfig(provider="mock", model="mock-urdu-bot")
        run_cfg = RunConfig(
            model=model_cfg,
            benchmark_id="urdu-qa",
            metrics=["exact_match", "f1"],
            use_cache=True,
            workers=1,
            max_samples=5,
            output_dir=tmpdir,
        )

        provider = MockProvider(
            model="mock-urdu-bot",
            canned_responses={"پاکستان": "اسلام آباد"},
            default_response="اسلام آباد",
            simulate_latency_ms=1.0,
        )

        runner = EvaluationRunner(
            config=run_cfg,
            benchmark=benchmark,
            provider=provider,
            cache_dir=Path(tmpdir) / ".cache",
        )

        run_result = runner.run(run_id="test_run_qa")

        assert run_result.run_id == "test_run_qa"
        assert run_result.total_samples == 5
        assert run_result.failed_samples == 0
        assert "exact_match" in run_result.metrics
        assert "f1" in run_result.metrics
        assert run_result.mean_latency_ms > 0.0

        # Verify output files persisted in output_dir / run_id
        run_dir = Path(tmpdir) / "test_run_qa"
        assert (run_dir / "config.json").exists()
        assert (run_dir / "scores.json").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "samples.jsonl").exists()

        # Check summary.json structure
        with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
            summary = json.load(f)
            assert summary["run_id"] == "test_run_qa"
            assert summary["total_samples"] == 5
            assert "metrics" in summary


def test_runner_multithreaded_execution() -> None:
    """Test parallel worker execution (workers=2)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        benchmark = get_benchmark("urdu-qa")
        model_cfg = ModelConfig(provider="mock", model="mock-parallel")
        run_cfg = RunConfig(
            model=model_cfg,
            benchmark_id="urdu-qa",
            metrics=["exact_match"],
            workers=2,
            max_samples=4,
            output_dir=tmpdir,
        )

        runner = EvaluationRunner(
            config=run_cfg,
            benchmark=benchmark,
            provider=MockProvider(model="mock-parallel", simulate_latency_ms=1.0),
            cache_dir=Path(tmpdir) / ".cache",
        )

        run_result = runner.run(run_id="test_parallel")
        assert run_result.total_samples == 4
        assert len(run_result.samples) == 4
