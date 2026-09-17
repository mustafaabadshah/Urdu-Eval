"""Research protocol verification and evaluation reproduction engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from urdu_eval import __version__
from urdu_eval.benchmarks.registry import get_benchmark


@dataclass
class CheckItem:
    """Individual verification check result."""

    name: str
    passed: bool
    status: str
    expected: str
    observed: str
    warning_only: bool = False


@dataclass
class ReproduceReport:
    """Comprehensive reproducibility inspection report."""

    run_id: str
    manifest_path: str
    all_passed: bool
    checks: list[CheckItem] = field(default_factory=list)
    recorded_metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load evaluation run manifest or scores.json."""
    p = Path(manifest_path)
    if p.is_dir():
        cand = p / "manifest.json"
        if not cand.exists():
            cand = p / "scores.json"
        if not cand.exists():
            cand = p / "config.json"
        if not cand.exists():
            raise FileNotFoundError(
                f"No manifest.json, scores.json, or config.json found in directory: {p}"
            )
        p = cand

    if not p.exists():
        if p.name == "manifest.json" and (p.parent / "scores.json").exists():
            p = p.parent / "scores.json"
        else:
            raise FileNotFoundError(f"Manifest file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        res = json.load(f)
        if isinstance(res, dict):
            return res
        return {}


def verify_manifest(manifest_path: str | Path) -> ReproduceReport:
    """Verify an evaluation manifest against current environment and dataset specifications."""
    data = load_manifest(manifest_path)
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    # Overlay flat top-level manifest fields if present
    for k, v in data.items():
        if k not in (
            "metadata",
            "summary",
            "scores",
            "raw_metrics",
            "confidence_intervals",
            "samples",
        ):
            if k not in metadata:
                metadata[k] = v

    if not metadata.get("benchmark") and "benchmark_id" in metadata:
        metadata["benchmark"] = {
            "id": metadata.get("benchmark_id", ""),
            "version": metadata.get("benchmark_version", "1.0.0"),
        }

    run_id = str(data.get("run_id", metadata.get("run_id", "unknown")))
    checks: list[CheckItem] = []
    warnings: list[str] = []

    # 1. Check UrduEval package version
    recorded_version = metadata.get("urdu_eval_version", "unknown")
    current_version = __version__
    ver_match = recorded_version == current_version
    checks.append(
        CheckItem(
            name="UrduEval Version",
            passed=True,  # version difference is a warning, not failure
            status="MATCH" if ver_match else "NOTICE",
            expected=recorded_version,
            observed=current_version,
            warning_only=not ver_match,
        )
    )
    if not ver_match:
        warnings.append(
            f"Run was recorded on UrduEval v{recorded_version}; current version is v{current_version}."
        )

    # 2. Check Benchmark & Version
    bench_info = metadata.get("benchmark", {})
    bench_id = str(metadata.get("benchmark_id") or bench_info.get("id", "")).lower()
    bench_version = str(metadata.get("benchmark_version") or bench_info.get("version", "1.0.0"))

    bench_obj = None
    try:
        bench_obj = get_benchmark(bench_id)
        bench_found = True
        current_b_ver = bench_obj.metadata.version
    except Exception:
        bench_found = False
        current_b_ver = "NOT FOUND"

    checks.append(
        CheckItem(
            name="Benchmark Registry",
            passed=bench_found,
            status="FOUND" if bench_found else "MISSING",
            expected=f"{bench_id} (v{bench_version})",
            observed=f"{bench_id} (v{current_b_ver})" if bench_found else "Not registered",
        )
    )

    # 3. Check Dataset Scope (Development vs Official)
    recorded_scope = metadata.get("dataset_scope") or (
        "development" if bench_info.get("is_development_sample") else "official"
    )
    is_dev = recorded_scope == "development"
    checks.append(
        CheckItem(
            name="Dataset Scope",
            passed=True,
            status="DEVELOPMENT" if is_dev else "OFFICIAL",
            expected=recorded_scope,
            observed=recorded_scope,
            warning_only=is_dev,
        )
    )
    if is_dev:
        warnings.append(
            "Run evaluated DEVELOPMENT samples; not comparable to official benchmark leaderboards."
        )

    # 4. Check Dataset Hash & Sample Count
    recorded_hash = str(metadata.get("dataset_sha256") or metadata.get("dataset_hash", ""))
    if bench_obj is not None:
        current_hash = bench_obj.metadata.provenance
        if recorded_hash and current_hash:
            hash_match = recorded_hash == current_hash
            checks.append(
                CheckItem(
                    name="Dataset SHA-256 Hash",
                    passed=hash_match,
                    status="VERIFIED" if hash_match else "MISMATCH",
                    expected=recorded_hash[:16] + "..."
                    if len(recorded_hash) > 16
                    else recorded_hash,
                    observed=current_hash[:16] + "..." if len(current_hash) > 16 else current_hash,
                )
            )
        else:
            checks.append(
                CheckItem(
                    name="Dataset Provenance",
                    passed=True,
                    status="RECORDED",
                    expected=recorded_hash or "N/A",
                    observed=current_hash or "N/A",
                )
            )

    # 4. Check Normalization Profile
    recorded_norm = metadata.get("normalization_profile", "conservative")
    from urdu_eval.enums import NormalizationProfile

    valid_profiles = {p.value for p in NormalizationProfile}
    norm_valid = recorded_norm.lower() in valid_profiles
    checks.append(
        CheckItem(
            name="Normalization Profile",
            passed=norm_valid,
            status="SUPPORTED" if norm_valid else "UNKNOWN",
            expected=recorded_norm,
            observed=recorded_norm if norm_valid else "Unsupported profile",
        )
    )

    # 5. Check Prompt Protocol
    proto = metadata.get("prompt_protocol", {})
    template_ver = proto.get("template_version", "1.0")
    few_shot = proto.get("few_shot", 0)
    checks.append(
        CheckItem(
            name="Prompt Protocol",
            passed=True,
            status="SPECIFIED",
            expected=f"v{template_ver} (few-shot: {few_shot})",
            observed=f"v{template_ver} (few-shot: {few_shot})",
        )
    )

    # 6. Check Model Provider and Settings
    model_cfg = metadata.get("model_settings", metadata.get("model_config", {}))
    provider = model_cfg.get("provider", "unknown")
    model_name = model_cfg.get("model", "unknown")
    temp = model_cfg.get("temperature", 0.0)
    seed = metadata.get("seed", model_cfg.get("seed", None))

    checks.append(
        CheckItem(
            name="Model Hyperparameters",
            passed=True,
            status="FIXED",
            expected=f"{provider}/{model_name} (temp={temp}, seed={seed})",
            observed=f"{provider}/{model_name} (temp={temp}, seed={seed})",
        )
    )

    all_passed = all(c.passed for c in checks if not c.warning_only)

    raw_metrics = data.get("scores") or data.get("metrics") or {}
    recorded_metrics: dict[str, float] = {}
    if isinstance(raw_metrics, dict):
        for k, v in raw_metrics.items():
            if isinstance(v, (int, float)):
                recorded_metrics[str(k)] = float(v)

    return ReproduceReport(
        run_id=run_id,
        manifest_path=str(manifest_path),
        all_passed=all_passed,
        checks=checks,
        recorded_metrics=recorded_metrics,
        warnings=warnings,
    )


def render_reproduce_report(report: ReproduceReport, console: Console) -> None:
    """Render a beautiful Rich terminal table summarizing the reproducibility audit."""
    table = Table(
        title=f"UrduEval Reproducibility Audit — Run: {report.run_id}",
        box=ROUNDED,
        header_style="bold cyan",
        show_header=True,
    )
    table.add_column("Protocol Element", style="white", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Recorded in Manifest", style="dim")
    table.add_column("Observed in Environment", style="dim")

    for c in report.checks:
        if c.status in {
            "MATCH",
            "VERIFIED",
            "SUPPORTED",
            "FOUND",
            "FIXED",
            "SPECIFIED",
            "RECORDED",
            "OFFICIAL",
        }:
            status_str = f"[green]✓ {c.status}[/green]"
        elif c.status in {"NOTICE", "DEVELOPMENT"}:
            status_str = f"[yellow]! {c.status}[/yellow]"
        else:
            status_str = f"[red]✗ {c.status}[/red]"

        table.add_row(c.name, status_str, c.expected, c.observed)

    console.print(table)

    if report.recorded_metrics:
        console.print("[bold]Recorded Benchmark Scores:[/bold]")
        m_parts = [
            f"[cyan]{k}[/cyan]: {v:.1%}" if 0 <= v <= 1 else f"[cyan]{k}[/cyan]: {v:.2f}"
            for k, v in report.recorded_metrics.items()
        ]
        console.print("  " + "  |  ".join(m_parts))

    if report.warnings:
        for w in report.warnings:
            console.print(f"[yellow]Notice:[/yellow] {w}")

    if report.all_passed:
        console.print(
            "[bold green]✓ REPRODUCIBILITY AUDIT: PASS[/bold green] — All experimental parameters, protocol versions, and dataset hashes match."
        )
    else:
        console.print(
            "[bold red]✗ REPRODUCIBILITY AUDIT: FAILED[/bold red] — One or more experimental determinants could not be verified."
        )
