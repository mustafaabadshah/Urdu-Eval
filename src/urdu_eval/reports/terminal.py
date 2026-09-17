"""Rich terminal formatting for beautiful CLI presentation."""

from __future__ import annotations

from typing import Any

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from urdu_eval import __version__
from urdu_eval.models import BenchmarkMetadata, RunResult, SampleResult

console = Console()


def render_banner() -> None:
    """Render the UrduEval welcome banner."""
    content = (
        "[bold cyan]UrduEval[/bold cyan] "
        f"[dim]v{__version__}[/dim]\n"
        "[white]Measure how well AI understands Urdu.[/white]\n"
        "[dim]Open evaluation infrastructure for Urdu & Roman Urdu AI[/dim]"
    )
    console.print(
        Panel(
            content,
            title="[bold green]اردو اِیوَل[/bold green]",
            subtitle="[dim]https://github.com/urdu-eval/urdu-eval[/dim]",
            box=ROUNDED,
            border_style="cyan",
            expand=False,
            padding=(1, 3),
        )
    )


def render_run_summary(run_result: RunResult) -> None:
    """Render rich terminal summary table for an evaluation run."""
    meta = run_result.metadata

    console.print()
    meta_table = Table(box=ROUNDED, show_header=False, border_style="dim")
    meta_table.add_column("Field", style="bold cyan")
    meta_table.add_column("Value", style="white")

    meta_table.add_row("Model", f"[bold green]{meta.model_settings.model}[/bold green]")
    meta_table.add_row("Provider", meta.model_settings.provider)
    meta_table.add_row("Benchmark", f"{meta.benchmark.name} ([dim]{meta.benchmark.id}[/dim])")
    meta_table.add_row("Total Samples", str(run_result.total_samples))
    meta_table.add_row("Mean Latency", f"{run_result.mean_latency_ms:.1f} ms")
    meta_table.add_row("Run ID", f"[yellow]{run_result.run_id}[/yellow]")
    console.print(meta_table)

    # Metrics Table
    console.print()
    metric_table = Table(
        title="Evaluation Metrics",
        box=ROUNDED,
        border_style="cyan",
        header_style="bold magenta",
    )
    metric_table.add_column("Metric", style="bold white")
    metric_table.add_column("Score", justify="right", style="bold green")
    metric_table.add_column("Percentage", justify="right", style="cyan")

    for m_name, score in run_result.metrics.items():
        pct = f"{score * 100:.1f}%" if 0.0 <= score <= 1.0 else "N/A"
        metric_table.add_row(m_name, f"{score:.4f}", pct)
    console.print(metric_table)

    # Failure Analysis Table
    if run_result.failure_summary:
        console.print()
        fail_table = Table(
            title="Error & Failure Analysis",
            box=ROUNDED,
            border_style="yellow",
            header_style="bold yellow",
        )
        fail_table.add_column("Category", style="white")
        fail_table.add_column("Count", justify="right", style="bold yellow")
        fail_table.add_column("Share", justify="right", style="dim")

        total = run_result.total_samples or 1
        for cat, count in sorted(run_result.failure_summary.items(), key=lambda x: -x[1]):
            pct = f"{(count / total) * 100:.1f}%"
            style = "green" if cat == "correct" else "red"
            fail_table.add_row(f"[{style}]{cat}[/{style}]", str(count), pct)
        console.print(fail_table)


def render_benchmarks_table(benchmarks: list[BenchmarkMetadata]) -> None:
    """Render list of available benchmarks."""
    table = Table(
        title="Available Benchmarks",
        box=ROUNDED,
        border_style="cyan",
        header_style="bold magenta",
    )
    table.add_column("ID", style="bold cyan")
    table.add_column("Name", style="white")
    table.add_column("Tasks", style="yellow")
    table.add_column("Scripts", style="dim")
    table.add_column("Metrics", style="green")
    table.add_column("Type", style="dim")

    for b in benchmarks:
        tasks = ", ".join(t.value for t in b.tasks)
        scripts = ", ".join(s.value for s in b.scripts)
        metrics = ", ".join(b.metrics)
        b_type = (
            "[yellow]Dev Sample[/yellow]" if b.is_development_sample else "[green]Official[/green]"
        )
        table.add_row(b.id, b.name, tasks, scripts, metrics, b_type)

    console.print(table)


def render_providers_table(providers: dict[str, dict[str, Any]]) -> None:
    """Render list of model providers and their environment status."""
    table = Table(
        title="Model Providers",
        box=ROUNDED,
        border_style="cyan",
        header_style="bold magenta",
    )
    table.add_column("Provider", style="bold cyan")
    table.add_column("Class", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Note / Error", style="dim")

    for name, info in providers.items():
        status = (
            "[bold green]✓ Available[/bold green]"
            if info["available"]
            else "[bold red]✗ Not Configured[/bold red]"
        )
        note = info.get("error") or "Ready to use"
        table.add_row(name, info["class"], status, note)

    console.print(table)


def render_comparison_table(results: list[RunResult]) -> None:
    """Render side-by-side comparison table for multiple model runs."""
    if not results:
        console.print("[yellow]No runs provided for comparison.[/yellow]")
        return

    table = Table(
        title="Model Comparison",
        box=ROUNDED,
        border_style="cyan",
        header_style="bold magenta",
    )
    table.add_column("Model", style="bold green")
    table.add_column("Provider", style="cyan")
    table.add_column("Benchmark", style="yellow")

    all_metric_names = sorted({m for r in results for m in r.metrics.keys()})
    for m in all_metric_names:
        table.add_column(m, justify="right", style="bold white")
    table.add_column("Latency", justify="right", style="dim")

    for r in results:
        row = [
            r.metadata.model_settings.model,
            r.metadata.model_settings.provider,
            r.metadata.benchmark.id,
        ]
        for m in all_metric_names:
            val = r.metrics.get(m)
            row.append(f"{val:.3f}" if val is not None else "-")
        row.append(f"{r.mean_latency_ms:.0f} ms")
        table.add_row(*row)

    console.print(table)


def render_sample_inspection(
    samples: list[SampleResult], failed_only: bool = False, limit: int = 10
) -> None:
    """Render individual sample details for error inspection."""
    shown = 0
    for s in samples:
        if failed_only and (s.failure_category and s.failure_category.value == "correct"):
            continue
        shown += 1
        if shown > limit:
            console.print(f"[dim]... and {len(samples) - limit} more samples[/dim]")
            break

        cat = s.failure_category.value if s.failure_category else "error"
        style = "green" if cat == "correct" else "red"

        content = (
            f"[bold cyan]Prompt:[/bold cyan] {s.prompt}\n"
            f"[bold green]Reference:[/bold green] {s.reference}\n"
            f"[bold magenta]Prediction:[/bold magenta] {s.prediction}\n"
            f"[bold]Category:[/bold] [{style}]{cat}[/{style}] | "
            f"[bold]Metrics:[/bold] {s.metrics} | "
            f"[bold]Latency:[/bold] {s.latency_ms:.1f}ms"
        )
        if s.error:
            content += f"\n[bold red]Error:[/bold red] {s.error}"

        console.print(
            Panel(
                content,
                title=f"Sample [bold]{s.sample_id}[/bold]",
                border_style=style,
                box=ROUNDED,
            )
        )
