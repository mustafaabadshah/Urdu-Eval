"""Leaderboard aggregator scanning evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

from urdu_eval.models import LeaderboardEntry, RunResult


def build_leaderboard(
    results_dir: str | Path,
    benchmark_id: str | None = None,
    task: str | None = None,
    script: str | None = None,
) -> list[LeaderboardEntry]:
    """Scan results directory, collect scores.json runs, and build leaderboard entries."""
    path = Path(results_dir)
    if not path.exists():
        return []

    entries: list[LeaderboardEntry] = []

    # Find all scores.json files recursively
    for score_file in path.glob("**/scores.json"):
        try:
            with score_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                run_res = RunResult.model_validate(data)

            # Filtering
            if benchmark_id and run_res.metadata.benchmark.id.lower() != benchmark_id.lower():
                continue

            if task:
                task_match = any(t.value == task.lower() for t in run_res.metadata.benchmark.tasks)
                if not task_match:
                    continue

            if script:
                script_match = any(
                    s.value == script.lower() for s in run_res.metadata.benchmark.scripts
                )
                if not script_match:
                    continue

            entry = LeaderboardEntry(
                model=run_res.metadata.model_settings.model,
                provider=run_res.metadata.model_settings.provider,
                benchmark_id=run_res.metadata.benchmark.id,
                benchmark_version=run_res.metadata.benchmark.version,
                samples=run_res.total_samples,
                metrics=run_res.metrics,
                latency_ms=run_res.mean_latency_ms,
                date=run_res.metadata.timestamp[:10],
                run_id=run_res.run_id,
            )
            entries.append(entry)
        except Exception:
            continue

    # Sort entries by primary metric (e.g., exact_match, accuracy, or f1) descending
    def sort_key(e: LeaderboardEntry) -> float:
        for metric_name in ["exact_match", "accuracy", "f1", "bleu", "rouge-l"]:
            if metric_name in e.metrics:
                return e.metrics[metric_name]
        return 0.0

    entries.sort(key=sort_key, reverse=True)
    return entries
