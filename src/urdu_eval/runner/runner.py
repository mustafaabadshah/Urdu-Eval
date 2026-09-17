"""Evaluation runner orchestrating models, benchmarks, metrics, and persistence."""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path

from urdu_eval import __version__
from urdu_eval.analysis.failures import categorize_failure
from urdu_eval.benchmarks.base import Benchmark
from urdu_eval.enums import TaskType
from urdu_eval.metrics.base import Metric, get_metric
from urdu_eval.models import (
    ModelResponse,
    RunConfig,
    RunMetadata,
    RunResult,
    Sample,
    SampleResult,
)
from urdu_eval.providers.base import ModelProvider, get_provider
from urdu_eval.runner.cache import EvaluationCache, compute_cache_key
from urdu_eval.runner.checkpoint import CheckpointManager
from urdu_eval.runner.retry import execute_with_retry


class EvaluationRunner:
    """Orchestrates model calls, metric evaluation, concurrency, and persistence."""

    def __init__(
        self,
        config: RunConfig,
        benchmark: Benchmark,
        provider: ModelProvider | None = None,
        metrics: list[Metric] | None = None,
        cache_dir: str | Path = ".urdu_eval_cache",
        on_sample_complete: Callable[[SampleResult], None] | None = None,
    ) -> None:
        self.config = config
        self.benchmark = benchmark
        self.provider = provider or get_provider(
            config.model.provider,
            model=config.model.model,
            **config.model.extra_params,
        )

        # Load metrics
        if metrics is not None:
            self.metrics = metrics
        else:
            metric_names = (
                config.metrics or self.benchmark.metadata.metrics or ["exact_match", "f1"]
            )
            self.metrics = [get_metric(m) for m in metric_names]

        self.cache = EvaluationCache(cache_dir=cache_dir) if config.use_cache else None
        self.on_sample_complete = on_sample_complete

    def _generate_response(self, sample: Sample) -> ModelResponse:
        """Fetch model response with optional cache and retry."""
        cache_key = None
        if self.cache is not None:
            cache_key = compute_cache_key(
                provider=self.config.model.provider,
                model=self.config.model.model,
                prompt=sample.prompt,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                top_p=self.config.model.top_p,
                seed=self.config.model.seed,
                system_prompt=self.config.model.system_prompt,
                benchmark_id=self.benchmark.metadata.id,
                benchmark_version=self.benchmark.metadata.version,
                prompt_template_version=self.config.prompt_protocol.template_version,
                extra_params=self.config.model.extra_params,
            )
            cached_resp = self.cache.get(cache_key)
            if cached_resp is not None:
                return cached_resp

        def _call() -> ModelResponse:
            return self.provider.generate(
                sample.prompt,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                system_prompt=self.config.model.system_prompt,
                **self.config.model.extra_params,
            )

        resp = execute_with_retry(_call, max_retries=3)

        if self.cache is not None and cache_key is not None:
            self.cache.set(cache_key, resp)

        return resp

    def _evaluate_single_sample(self, sample: Sample) -> SampleResult:
        """Evaluate a single sample through model and all metrics."""
        try:
            resp = self._generate_response(sample)
            prediction = resp.text
            latency_ms = resp.latency_ms
            error = None
        except Exception as exc:
            prediction = ""
            latency_ms = 0.0
            error = str(exc)

        # Compute metric scores
        metric_scores: dict[str, float] = {}
        if error is None:
            from urdu_eval.metrics.mcq import extract_mcq_choice

            eval_pred = prediction
            is_mcq = sample.task in {TaskType.MMLU, TaskType.MCQA, TaskType.MINIMAL_PAIR} or bool(
                sample.options
            )
            if is_mcq and isinstance(sample.reference, str) and len(sample.reference.strip()) <= 2:
                extracted = extract_mcq_choice(prediction, options=sample.options)
                if extracted is not None:
                    eval_pred = extracted

            for m in self.metrics:
                try:
                    score = m.compute(
                        prediction=eval_pred,
                        reference=sample.reference,
                        prompt=sample.prompt,
                    )
                    metric_scores[m.name] = score
                except Exception:
                    metric_scores[m.name] = 0.0

        # Categorize failure
        cat = categorize_failure(sample, prediction, metric_scores) if error is None else None

        result = SampleResult(
            sample_id=sample.id,
            prompt=sample.prompt,
            reference=sample.reference,
            prediction=prediction,
            metrics=metric_scores,
            latency_ms=latency_ms,
            failure_category=cat,
            error=error,
        )

        if self.on_sample_complete is not None:
            self.on_sample_complete(result)

        return result

    def run(self, run_id: str | None = None) -> RunResult:
        """Execute the complete evaluation run."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        final_run_id = (
            run_id
            or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.benchmark.metadata.id}"
        )

        output_dir = Path(self.config.output_dir) / final_run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_mgr = CheckpointManager(output_dir)

        # Save config immediately
        config_path = output_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(self.config.model_dump(mode="json"), f, indent=2)

        # Collect samples to evaluate
        all_samples = list(self.benchmark.load_samples(max_samples=self.config.max_samples))
        pending_samples = [s for s in all_samples if not checkpoint_mgr.is_completed(s.id)]

        # Execute samples
        if pending_samples:
            if self.config.workers > 1:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config.workers
                ) as executor:
                    futures = {
                        executor.submit(self._evaluate_single_sample, s): s for s in pending_samples
                    }
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        checkpoint_mgr.record_sample(res)
            else:
                for s in pending_samples:
                    res = self._evaluate_single_sample(s)
                    checkpoint_mgr.record_sample(res)

        # Reassemble all sample results in original order
        completed_results_map = {r.sample_id: r for r in checkpoint_mgr.get_completed_results()}
        ordered_sample_results: list[SampleResult] = [
            completed_results_map[s.id] for s in all_samples if s.id in completed_results_map
        ]

        # Compute summary metrics
        total = len(ordered_sample_results)
        failed_count = sum(1 for r in ordered_sample_results if r.error is not None)
        mean_latency = (
            sum(r.latency_ms for r in ordered_sample_results) / total if total > 0 else 0.0
        )

        from urdu_eval.metrics.stats import compute_metric_ci

        aggregated_metrics: dict[str, float] = {}
        confidence_intervals: dict[str, tuple[float, float]] = {}
        for m in self.metrics:
            valid_scores = [
                r.metrics[m.name]
                for r in ordered_sample_results
                if m.name in r.metrics and r.error is None
            ]
            mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            aggregated_metrics[m.name] = mean_score
            confidence_intervals[m.name] = compute_metric_ci(
                m.name,
                valid_scores,
                confidence=self.config.ci_config.confidence,
                method=self.config.ci_config.method,
                resamples=self.config.ci_config.resamples,
                seed=self.config.ci_config.seed,
            )

        # Compute raw unnormalized exact match for transparency
        raw_em_scores = [
            1.0 if r.prediction.strip() == str(r.reference).strip() else 0.0
            for r in ordered_sample_results
            if r.error is None
        ]
        raw_metrics: dict[str, float] = {
            "raw_exact_match": sum(raw_em_scores) / len(raw_em_scores) if raw_em_scores else 0.0
        }

        # Compute failure breakdown
        failure_summary: dict[str, int] = {}
        for r in ordered_sample_results:
            if r.failure_category is not None:
                cat_name = str(r.failure_category.value)
                failure_summary[cat_name] = failure_summary.get(cat_name, 0) + 1
            elif r.error is not None:
                failure_summary["error"] = failure_summary.get("error", 0) + 1

        run_metadata = RunMetadata(
            run_id=final_run_id,
            timestamp=timestamp,
            urdu_eval_version=__version__,
            benchmark=self.benchmark.metadata,
            dataset_hash=self.benchmark.metadata.provenance,
            model_config=self.config.model,
            normalization_profile=self.config.normalization_profile,
            python_version=sys.version.split()[0],
            platform=f"{platform.system()} {platform.release()}",
            seed=getattr(self.config.model, "seed", None),
            top_p=getattr(self.config.model, "top_p", None),
            prompt_protocol=self.config.prompt_protocol,
            contamination=self.config.contamination,
            ci_config=self.config.ci_config,
        )

        run_result = RunResult(
            run_id=final_run_id,
            metadata=run_metadata,
            metrics=aggregated_metrics,
            raw_metrics=raw_metrics,
            confidence_intervals=confidence_intervals,
            ci_config=self.config.ci_config,
            samples=ordered_sample_results,
            failure_summary=failure_summary,
            total_samples=total,
            failed_samples=failed_count,
            mean_latency_ms=mean_latency,
        )

        # Write results files
        self._save_results(output_dir, run_result)

        return run_result

    def _save_results(self, output_dir: Path, run_result: RunResult) -> None:
        """Persist scores.json, summary.json, and samples.jsonl."""
        # 1. scores.json and manifest.json (full serializable object)
        result_json = run_result.model_dump_json(indent=2)
        with (output_dir / "scores.json").open("w", encoding="utf-8") as f:
            f.write(result_json)
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
            f.write(result_json)

        # 2. summary.json (compact top-level overview)
        summary = {
            "run_id": run_result.run_id,
            "model": run_result.metadata.model_settings.model,
            "provider": run_result.metadata.model_settings.provider,
            "benchmark": run_result.metadata.benchmark.id,
            "metrics": run_result.metrics,
            "total_samples": run_result.total_samples,
            "failed_samples": run_result.failed_samples,
            "mean_latency_ms": run_result.mean_latency_ms,
            "failure_summary": run_result.failure_summary,
            "timestamp": run_result.metadata.timestamp,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 3. samples.jsonl
        with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
            for s in run_result.samples:
                f.write(s.model_dump_json() + "\n")
