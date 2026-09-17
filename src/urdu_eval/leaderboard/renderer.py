"""Leaderboard exporter for Table, Markdown, CSV, JSON, and HTML."""

from __future__ import annotations

import csv
import io
import json

from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from urdu_eval.models import LeaderboardEntry


def render_leaderboard_terminal(entries: list[LeaderboardEntry]) -> None:
    """Render leaderboard to rich terminal table."""
    console = Console()
    if not entries:
        console.print("[yellow]No leaderboard entries found for the given criteria.[/yellow]")
        return

    table = Table(
        title="UrduEval Leaderboard",
        box=ROUNDED,
        border_style="cyan",
        header_style="bold magenta",
    )
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Model", style="bold green")
    table.add_column("Provider", style="cyan")
    table.add_column("Benchmark", style="yellow")
    table.add_column("Samples", justify="right", style="dim")

    all_metrics = sorted({m for e in entries for m in e.metrics.keys()})
    for m in all_metrics:
        table.add_column(m.upper(), justify="right", style="bold white")

    table.add_column("Latency", justify="right", style="dim")
    table.add_column("Date", style="dim")

    for rank, e in enumerate(entries, start=1):
        row = [
            str(rank),
            e.model,
            e.provider,
            f"{e.benchmark_id} (v{e.benchmark_version})",
            str(e.samples),
        ]
        for m in all_metrics:
            val = e.metrics.get(m)
            row.append(
                f"{val * 100:.1f}%"
                if val is not None and 0.0 <= val <= 1.0
                else f"{val:.3f}"
                if val is not None
                else "-"
            )
        row.append(f"{e.latency_ms:.0f}ms")
        row.append(e.date)
        table.add_row(*row)

    console.print(table)


def export_leaderboard_markdown(entries: list[LeaderboardEntry]) -> str:
    """Export leaderboard as GitHub Markdown table."""
    if not entries:
        return "*No evaluation runs recorded in leaderboard.*"

    all_metrics = sorted({m for e in entries for m in e.metrics.keys()})
    headers = (
        ["Rank", "Model", "Provider", "Benchmark", "Samples"]
        + [m.upper() for m in all_metrics]
        + ["Latency", "Date"]
    )

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---:"] + [":---"] * 3 + [":---:"] * (len(all_metrics) + 2)) + " |",
    ]

    for rank, e in enumerate(entries, start=1):
        cols = [
            str(rank),
            f"`{e.model}`",
            e.provider,
            f"`{e.benchmark_id}`",
            str(e.samples),
        ]
        for m in all_metrics:
            val = e.metrics.get(m)
            cols.append(f"{val * 100:.1f}%" if val is not None and 0.0 <= val <= 1.0 else "-")
        cols.append(f"{e.latency_ms:.0f}ms")
        cols.append(e.date)
        lines.append("| " + " | ".join(cols) + " |")

    return "\n".join(lines)


def export_leaderboard_csv(entries: list[LeaderboardEntry]) -> str:
    """Export leaderboard as CSV string."""
    output = io.StringIO()
    all_metrics = sorted({m for e in entries for m in e.metrics.keys()})
    headers = (
        ["rank", "model", "provider", "benchmark_id", "benchmark_version", "samples"]
        + all_metrics
        + ["latency_ms", "date", "run_id"]
    )

    writer = csv.writer(output)
    writer.writerow(headers)

    for rank, e in enumerate(entries, start=1):
        row = [rank, e.model, e.provider, e.benchmark_id, e.benchmark_version, e.samples]
        for m in all_metrics:
            row.append(e.metrics.get(m, ""))
        row.extend([e.latency_ms, e.date, e.run_id])
        writer.writerow(row)

    return output.getvalue()


def export_leaderboard_json(entries: list[LeaderboardEntry]) -> str:
    """Export leaderboard as JSON string."""
    return json.dumps([e.model_dump(mode="json") for e in entries], indent=2)
