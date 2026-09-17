"""Report generators for UrduEval (Terminal, JSON, Markdown, HTML)."""

from urdu_eval.reports.html import generate_html_report
from urdu_eval.reports.json import export_json_report
from urdu_eval.reports.markdown import generate_markdown_report
from urdu_eval.reports.terminal import (
    render_banner,
    render_benchmarks_table,
    render_comparison_table,
    render_providers_table,
    render_run_summary,
    render_sample_inspection,
)

__all__ = [
    "render_banner",
    "render_run_summary",
    "render_benchmarks_table",
    "render_providers_table",
    "render_comparison_table",
    "render_sample_inspection",
    "generate_markdown_report",
    "generate_html_report",
    "export_json_report",
]
