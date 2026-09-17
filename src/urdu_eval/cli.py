"""Command Line Interface for UrduEval."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import typer
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from urdu_eval import __version__
from urdu_eval.benchmarks import (
    Benchmark,
    CustomBenchmark,
    get_benchmark,
    list_benchmarks,
)
from urdu_eval.dataset import validate_dataset
from urdu_eval.leaderboard import (
    build_leaderboard,
    export_leaderboard_csv,
    export_leaderboard_json,
    export_leaderboard_markdown,
    render_leaderboard_terminal,
)
from urdu_eval.metrics import list_metrics
from urdu_eval.models import ModelConfig, RunConfig, RunResult
from urdu_eval.providers import list_providers
from urdu_eval.reports import (
    generate_html_report,
    generate_markdown_report,
    render_banner,
    render_benchmarks_table,
    render_comparison_table,
    render_providers_table,
    render_run_summary,
    render_sample_inspection,
)
from urdu_eval.runner import EvaluationRunner

app = typer.Typer(
    name="urdu-eval",
    help="UrduEval: Open evaluation infrastructure for Urdu & Roman Urdu AI.",
    add_completion=False,
)
console = Console()


@app.command(name="version")
def version_command() -> None:
    """Show UrduEval version, environment, and system details."""
    render_banner()
    console.print(f"[bold cyan]UrduEval Version:[/bold cyan] {__version__}")
    console.print(f"[bold cyan]Python Version:[/bold cyan]   {sys.version.split()[0]}")
    console.print(
        f"[bold cyan]Platform:[/bold cyan]         {platform.system()} {platform.release()}"
    )


@app.command(name="benchmarks")
def benchmarks_command() -> None:
    """List all registered and built-in benchmark suites."""
    benchmarks = list_benchmarks()
    render_benchmarks_table(benchmarks)


@app.command(name="providers")
def providers_command() -> None:
    """List available model providers and their environment status."""
    providers = list_providers()
    render_providers_table(providers)


@app.command(name="metrics")
def metrics_command() -> None:
    """List all supported evaluation metrics."""
    metrics = list_metrics()
    console.print("[bold cyan]Available Evaluation Metrics:[/bold cyan]")
    for m in metrics:
        console.print(f"  • [bold green]{m}[/bold green]")


@app.command(name="validate")
def validate_command(
    dataset_path: Path = typer.Argument(..., help="Path to JSONL dataset to validate"),
) -> None:
    """Validate a JSONL benchmark dataset for schema and linguistic correctness."""
    console.print(f"[dim]Validating dataset:[/dim] {dataset_path}")
    result = validate_dataset(dataset_path)

    if result.is_valid:
        console.print("[bold green]✓ Dataset is valid.[/bold green]")
        console.print(result.summary())
        raise typer.Exit(code=0)

    console.print("[bold red]✗ Dataset validation failed:[/bold red]")
    console.print(result.summary())
    raise typer.Exit(code=1)


@app.command(name="run")
def run_command(
    model: str = typer.Option("mock-urdu-model", "--model", "-m", help="Model name or identifier"),
    provider: str = typer.Option(
        "mock",
        "--provider",
        "-p",
        help="Provider (mock, openai, anthropic, ollama, hf, http, openrouter)",
    ),
    benchmark: str | None = typer.Option(
        None, "--benchmark", "-b", help="Registered benchmark ID (e.g. urdu-qa)"
    ),
    dataset: Path | None = typer.Option(None, "--dataset", "-d", help="Custom JSONL dataset path"),
    metrics: str = typer.Option("exact_match,f1", "--metrics", help="Comma-separated metric names"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of concurrent worker threads"),
    cache: bool = typer.Option(
        True, "--cache/--no-cache", help="Enable or disable response caching"
    ),
    resume: str | None = typer.Option(
        None, "--resume", help="Resume previous evaluation by Run ID"
    ),
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-n", help="Limit number of samples"
    ),
    output_dir: str = typer.Option("results", "--output", "-o", help="Directory for run results"),
) -> None:
    """Run an evaluation benchmark against a model."""
    render_banner()

    # Resolve benchmark
    bm: Benchmark
    if dataset is not None:
        try:
            bm = CustomBenchmark(dataset)
        except Exception as exc:
            console.print(f"[bold red]Error loading custom dataset:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc
    elif benchmark is not None:
        try:
            bm = get_benchmark(benchmark)
        except ValueError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc
    else:
        console.print(
            "[bold red]Error:[/bold red] You must specify either --benchmark or --dataset."
        )
        raise typer.Exit(code=1)

    parsed_metrics = [m.strip() for m in metrics.split(",") if m.strip()]
    model_cfg = ModelConfig(provider=provider, model=model)
    run_cfg = RunConfig(
        model=model_cfg,
        benchmark_id=bm.metadata.id,
        dataset_path=str(dataset) if dataset else None,
        metrics=parsed_metrics,
        use_cache=cache,
        workers=workers,
        max_samples=max_samples,
        output_dir=output_dir,
    )

    runner = EvaluationRunner(config=run_cfg, benchmark=bm)

    with console.status(f"[bold cyan]Evaluating {model} on {bm.metadata.name}...[/bold cyan]"):
        try:
            run_result = runner.run(run_id=resume)
        except Exception as exc:
            console.print(f"[bold red]Evaluation failed:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

    render_run_summary(run_result)
    out_dir = Path(output_dir) / run_result.run_id
    console.print(f"\n[bold green]Results saved to:[/bold green] [underline]{out_dir}[/underline]")


@app.command(name="check")
def check_command(
    model: str = typer.Option("mock-urdu-model", "--model", "-m", help="Model name"),
    provider: str = typer.Option("mock", "--provider", "-p", help="Provider"),
) -> None:
    """Run a rapid diagnostic 'Urdu Health Check' across core Urdu capabilities."""
    render_banner()
    console.print(f"[bold cyan]Running Urdu Health Check for:[/bold cyan] {model} ({provider})\n")

    benchmarks_to_check = [
        ("Urdu comprehension", "urdu-qa"),
        ("Roman Urdu", "urdu-roman"),
        ("Translation", "urdu-translation"),
        ("Summarization", "urdu-summary"),
        ("Reasoning", "urdu-reasoning"),
        ("Domain Knowledge", "urdu-mmlu"),
    ]

    table = Table(box=ROUNDED, border_style="cyan", header_style="bold magenta")
    table.add_column("Capability", style="bold white")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    for label, b_id in benchmarks_to_check:
        try:
            bm = get_benchmark(b_id)
            m_cfg = ModelConfig(provider=provider, model=model)
            r_cfg = RunConfig(
                model=m_cfg,
                benchmark_id=b_id,
                metrics=["exact_match", "f1"],
                max_samples=2,
            )
            runner = EvaluationRunner(config=r_cfg, benchmark=bm)
            res = runner.run()
            status = (
                "[bold green]✓ Operational[/bold green]"
                if res.failed_samples == 0
                else "[bold yellow]⚠ Issues Detected[/bold yellow]"
            )
            details = f"{res.total_samples} samples evaluated ({res.mean_latency_ms:.0f}ms avg)"
        except Exception as exc:
            status = "[bold red]✗ Failed[/bold red]"
            details = str(exc)

        table.add_row(label, status, details)

    console.print(table)


@app.command(name="compare")
def compare_command(
    result_files: list[Path] = typer.Argument(..., help="List of scores.json files to compare"),
) -> None:
    """Compare multiple model evaluation results side-by-side."""
    render_banner()
    results: list[RunResult] = []

    for path in result_files:
        if not path.exists():
            console.print(f"[yellow]Skipping missing file:[/yellow] {path}")
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                res = RunResult.model_validate(data)
                results.append(res)
        except Exception as exc:
            console.print(f"[red]Error parsing {path}:[/red] {exc}")

    if not results:
        console.print("[bold red]No valid run results to compare.[/bold red]")
        raise typer.Exit(code=1)

    render_comparison_table(results)


@app.command(name="report")
def report_command(
    results_file: Path = typer.Argument(..., help="Path to scores.json"),
    html_out: Path | None = typer.Option(None, "--html", help="Path to output HTML report"),
    markdown_out: Path | None = typer.Option(
        None, "--markdown", help="Path to output Markdown report"
    ),
) -> None:
    """Generate HTML, Markdown, or terminal reports from run results."""
    if not results_file.exists():
        console.print(f"[bold red]File not found:[/bold red] {results_file}")
        raise typer.Exit(code=1)

    with results_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        run_res = RunResult.model_validate(data)

    if html_out is not None:
        html_content = generate_html_report(run_res)
        html_out.parent.mkdir(parents=True, exist_ok=True)
        with html_out.open("w", encoding="utf-8") as f:
            f.write(html_content)
        console.print(f"[bold green]HTML report generated:[/bold green] {html_out}")

    if markdown_out is not None:
        md_content = generate_markdown_report(run_res)
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        with markdown_out.open("w", encoding="utf-8") as f:
            f.write(md_content)
        console.print(f"[bold green]Markdown report generated:[/bold green] {markdown_out}")

    if html_out is None and markdown_out is None:
        render_run_summary(run_res)


@app.command(name="inspect")
def inspect_command(
    results_file: Path = typer.Argument(..., help="Path to scores.json"),
    failed_only: bool = typer.Option(
        False, "--failed-only", "-f", help="Show only failed or incorrect samples"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of samples to display"),
) -> None:
    """Inspect sample-level prompts, references, predictions, and failure categories."""
    if not results_file.exists():
        console.print(f"[bold red]File not found:[/bold red] {results_file}")
        raise typer.Exit(code=1)

    with results_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        run_res = RunResult.model_validate(data)

    render_sample_inspection(run_res.samples, failed_only=failed_only, limit=limit)


@app.command(name="leaderboard")
def leaderboard_command(
    results_dir: Path = typer.Argument(
        Path("results"), help="Directory containing evaluation runs"
    ),
    benchmark: str | None = typer.Option(None, "--benchmark", "-b", help="Filter by benchmark ID"),
    task: str | None = typer.Option(None, "--task", "-t", help="Filter by task"),
    script: str | None = typer.Option(
        None, "--script", "-s", help="Filter by script (urdu, roman_urdu)"
    ),
    format_type: str = typer.Option(
        "table", "--format", help="Output format: table, markdown, csv, json"
    ),
) -> None:
    """Aggregate all evaluation runs and generate a comparative leaderboard."""
    entries = build_leaderboard(
        results_dir=results_dir,
        benchmark_id=benchmark,
        task=task,
        script=script,
    )

    fmt = format_type.lower()
    if fmt == "table":
        render_leaderboard_terminal(entries)
    elif fmt == "markdown":
        console.print(export_leaderboard_markdown(entries))
    elif fmt == "csv":
        console.print(export_leaderboard_csv(entries))
    elif fmt == "json":
        console.print(export_leaderboard_json(entries))
    else:
        console.print(f"[bold red]Unsupported format:[/bold red] {format_type}")
        raise typer.Exit(code=1)
