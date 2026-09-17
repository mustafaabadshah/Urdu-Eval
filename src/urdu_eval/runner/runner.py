"""Evaluation runner orchestrating models, benchmarks, metrics, and persistence."""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

        # Determine dataset scope and official evaluation status
        is_dev = (
            getattr(self.benchmark.metadata, "is_development_sample", False)
            or getattr(self.benchmark.metadata, "dataset_scope", "official") == "development"
        )
        dataset_scope = "development" if is_dev else "official"
        is_official = not is_dev
        official_size = getattr(self.benchmark.metadata, "official_benchmark_size", None)

        model_cfg = self.config.model
        prompt_proto = self.config.prompt_protocol
        ci_cfg = self.config.ci_config
        contam_info = self.config.contamination

        # Resolve real cryptographic SHA-256 for dataset bytes
        from urdu_eval.dataset import compute_dataset_hash

        real_dataset_sha256 = getattr(self.benchmark.metadata, "dataset_sha256", "")
        if not real_dataset_sha256 and self.config.dataset_path:
            p = Path(self.config.dataset_path)
            if p.exists() and p.is_file():
                real_dataset_sha256 = compute_dataset_hash(p)
        if not real_dataset_sha256 and hasattr(self.benchmark, "file_path"):
            p = self.benchmark.file_path
            if isinstance(p, Path) and p.exists() and p.is_file():
                real_dataset_sha256 = compute_dataset_hash(p)
        if not real_dataset_sha256:
            # Check dev_samples directory for registered benchmarks
            dev_p = (
                Path(__file__).parent.parent
                / "benchmarks"
                / "dev_samples"
                / f"{self.benchmark.metadata.id.replace('-', '_')}.jsonl"
            )
            if dev_p.exists():
                real_dataset_sha256 = compute_dataset_hash(dev_p)

        dataset_sha256 = real_dataset_sha256 or "N/A (Streaming/External)"
        dataset_hash = real_dataset_sha256 or self.benchmark.metadata.provenance

        run_metadata = RunMetadata(
            run_id=final_run_id,
            timestamp=timestamp,
            urdu_eval_version=__version__,
            benchmark=self.benchmark.metadata,
            benchmark_id=self.benchmark.metadata.id,
            benchmark_version=self.benchmark.metadata.version,
            dataset_hash=dataset_hash,
            dataset_sha256=dataset_sha256,
            dataset_source=self.benchmark.metadata.source,
            dataset_provenance=self.benchmark.metadata.provenance,
            dataset_size=total,
            dataset_scope=dataset_scope,
            is_official_evaluation=is_official,
            official_benchmark_size=official_size,
            model_config=model_cfg,
            model=model_cfg.model,
            provider=model_cfg.provider,
            model_revision=getattr(model_cfg, "revision", "default") or "default",
            temperature=getattr(model_cfg, "temperature", 0.0),
            top_p=getattr(model_cfg, "top_p", None),
            max_tokens=getattr(model_cfg, "max_tokens", None),
            seed=getattr(model_cfg, "seed", None),
            prompt_protocol=prompt_proto,
            prompt_template_version=getattr(prompt_proto, "template_version", "1.0"),
            prompt_language=getattr(prompt_proto, "language", "urdu"),
            shot_count=getattr(prompt_proto, "few_shot", 0),
            answer_extraction_version=getattr(prompt_proto, "answer_extraction_version", "mcq-v1"),
            normalization_profile=self.config.normalization_profile,
            normalization_version="v1.0",
            metrics=[m.name for m in self.metrics],
            ci_config=ci_cfg,
            ci_method=ci_cfg.method.value
            if hasattr(ci_cfg.method, "value")
            else str(ci_cfg.method),
            ci_resamples=ci_cfg.resamples,
            ci_seed=ci_cfg.seed,
            contamination=contam_info,
            contamination_status=contam_info.status.value
            if hasattr(contam_info.status, "value")
            else str(contam_info.status),
            python_version=sys.version.split()[0],
            platform=f"{platform.system()} {platform.release()}",
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
        """Persist scores.json, manifest.json, summary.json, and samples.jsonl."""
        # 1. scores.json (full serializable object with samples)
        result_json = run_result.model_dump_json(indent=2)
        with (output_dir / "scores.json").open("w", encoding="utf-8") as f:
            f.write(result_json)

        # 2. manifest.json (canonical research reproduction manifest)
        meta = run_result.metadata
        manifest_payload: dict[str, Any] = {
            "urdu_eval_version": meta.urdu_eval_version,
            "run_id": meta.run_id,
            "timestamp": meta.timestamp,
            "benchmark_id": meta.benchmark_id,
            "benchmark_version": meta.benchmark_version,
            "dataset_sha256": meta.dataset_sha256,
            "dataset_source": meta.dataset_source,
            "dataset_provenance": meta.dataset_provenance,
            "dataset_size": meta.dataset_size,
            "dataset_scope": meta.dataset_scope,
            "is_official_evaluation": meta.is_official_evaluation,
            "official_benchmark_size": meta.official_benchmark_size,
            "prompt_template_version": meta.prompt_template_version,
            "prompt_language": meta.prompt_language,
            "shot_count": meta.shot_count,
            "answer_extraction_version": meta.answer_extraction_version,
            "model": meta.model,
            "provider": meta.provider,
            "model_revision": meta.model_revision,
            "temperature": meta.temperature,
            "top_p": meta.top_p,
            "max_tokens": meta.max_tokens,
            "seed": meta.seed,
            "normalization_profile": meta.normalization_profile,
            "normalization_version": meta.normalization_version,
            "metrics": meta.metrics,
            "scores": run_result.metrics,
            "raw_metrics": run_result.raw_metrics,
            "confidence_intervals": run_result.confidence_intervals,
            "ci_method": meta.ci_method,
            "ci_resamples": meta.ci_resamples,
            "ci_seed": meta.ci_seed,
            "contamination_status": meta.contamination_status,
            "python_version": meta.python_version,
            "platform": meta.platform,
            "metadata": meta.model_dump(),
        }
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)

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
