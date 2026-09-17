"""Markdown report generation."""

from __future__ import annotations

from urdu_eval.models import RunResult


def generate_markdown_report(run_result: RunResult) -> str:
    """Generate a GitHub-flavored Markdown evaluation report."""
    meta = run_result.metadata
    lines: list[str] = [
        f"# UrduEval Evaluation Report: {meta.benchmark.name}",
        "",
        "## Overview",
        "",
        "| Field | Value |",
        "| :--- | :--- |",
        f"| **Model** | `{meta.model_settings.model}` |",
        f"| **Provider** | `{meta.model_settings.provider}` |",
        f"| **Benchmark** | `{meta.benchmark.name}` (`{meta.benchmark.id}`) |",
        f"| **UrduEval Version** | `{meta.urdu_eval_version}` |",
        f"| **Run ID** | `{run_result.run_id}` |",
        f"| **Timestamp** | `{meta.timestamp}` |",
        f"| **Total Samples** | `{run_result.total_samples}` |",
        f"| **Failed Samples** | `{run_result.failed_samples}` |",
        f"| **Mean Latency** | `{run_result.mean_latency_ms:.1f} ms` |",
        "",
        "## Benchmark Scores",
        "",
        "| Metric | Score | Percentage |",
        "| :--- | :---: | :---: |",
    ]

    for m_name, val in run_result.metrics.items():
        pct = f"{val * 100:.1f}%" if 0.0 <= val <= 1.0 else "N/A"
        lines.append(f"| **{m_name}** | `{val:.4f}` | {pct} |")

    lines.extend(
        [
            "",
            "## Failure & Linguistic Analysis",
            "",
            "| Category | Count | Proportion |",
            "| :--- | :---: | :---: |",
        ]
    )

    total = run_result.total_samples or 1
    for cat, cnt in sorted(run_result.failure_summary.items(), key=lambda x: -x[1]):
        pct = f"{(cnt / total) * 100:.1f}%"
        lines.append(f"| `{cat}` | {cnt} | {pct} |")

    lines.extend(
        [
            "",
            "## Sample Excerpts",
            "",
        ]
    )

    for i, s in enumerate(run_result.samples[:10], start=1):
        cat = s.failure_category.value if s.failure_category else "error"
        ref_text = s.reference if isinstance(s.reference, str) else ", ".join(s.reference)
        lines.extend(
            [
                f"### Sample {i} (`{s.sample_id}`)",
                f"- **Prompt:** {s.prompt}",
                f"- **Reference:** `{ref_text}`",
                f"- **Prediction:** `{s.prediction}`",
                f"- **Category:** `{cat}`",
                f"- **Metrics:** `{s.metrics}`",
                "",
            ]
        )

    return "\n".join(lines)
