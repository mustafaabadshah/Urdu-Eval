"""Tests for reproducibility verification engine."""

import json
from pathlib import Path

from rich.console import Console

from urdu_eval.runner.reproduce import (
    load_manifest,
    render_reproduce_report,
    verify_manifest,
)


def test_verify_manifest_mock(tmp_path: Path) -> None:
    """Verify manifest audit passes for a consistent run metadata file."""
    manifest_file = tmp_path / "scores.json"
    manifest_data = {
        "run_id": "test_repro_run",
        "metadata": {
            "run_id": "test_repro_run",
            "timestamp": "2026-09-17T22:00:00Z",
            "urdu_eval_version": "0.1.2",
            "benchmark": {
                "id": "urdu-qa",
                "name": "Urdu QA (Dev Sample)",
                "version": "0.1.0",
            },
            "dataset_hash": "",
            "model_settings": {
                "provider": "mock",
                "model": "mock-urdu-model",
                "temperature": 0.0,
                "seed": 42,
            },
            "normalization_profile": "conservative",
            "prompt_protocol": {
                "template_version": "1.0",
                "few_shot": 0,
            },
        },
        "metrics": {"exact_match": 1.0, "f1": 1.0},
    }

    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    report = verify_manifest(manifest_file)
    assert report.run_id == "test_repro_run"
    assert report.all_passed is True

    # Render test without crash
    console = Console(record=True)
    render_reproduce_report(report, console)
    out = console.export_text()
    assert "REPRODUCIBILITY AUDIT: PASS" in out


def test_load_manifest_dir(tmp_path: Path) -> None:
    """Verify load_manifest works when passed a run directory path."""
    scores_file = tmp_path / "scores.json"
    with scores_file.open("w", encoding="utf-8") as f:
        json.dump({"run_id": "dir_test"}, f)

    loaded = load_manifest(tmp_path)
    assert loaded["run_id"] == "dir_test"
