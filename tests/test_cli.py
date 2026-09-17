"""Tests for Typer CLI commands."""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from tests.test_reports import _create_mock_run_result
from urdu_eval.cli import app

runner = CliRunner(env={"COLUMNS": "240"})


def test_cli_version() -> None:
    """Test 'urdu-eval version' command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "UrduEval Version" in result.stdout


def test_cli_benchmarks() -> None:
    """Test 'urdu-eval benchmarks' command."""
    result = runner.invoke(app, ["benchmarks"])
    assert result.exit_code == 0
    assert "urdu-qa" in result.stdout
    assert "urdu-reasoning" in result.stdout


def test_cli_providers() -> None:
    """Test 'urdu-eval providers' command."""
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0
    assert "mock" in result.stdout
    assert "http" in result.stdout


def test_cli_metrics() -> None:
    """Test 'urdu-eval metrics' command."""
    result = runner.invoke(app, ["metrics"])
    assert result.exit_code == 0
    assert "exact_match" in result.stdout
    assert "f1" in result.stdout
    assert "bleu" in result.stdout


def test_cli_validate() -> None:
    """Test 'urdu-eval validate' command."""
    # Valid dataset
    valid_res = runner.invoke(app, ["validate", "examples/sample_qa.jsonl"])
    assert valid_res.exit_code == 0
    assert "Dataset is valid" in valid_res.stdout

    # Invalid dataset
    invalid_res = runner.invoke(app, ["validate", "examples/invalid_sample.jsonl"])
    assert invalid_res.exit_code == 1
    assert "validation failed" in invalid_res.stdout


def test_cli_run_mock() -> None:
    """Test 'urdu-eval run' command with mock provider."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(
            app,
            [
                "run",
                "--provider",
                "mock",
                "--benchmark",
                "urdu-qa",
                "--metrics",
                "exact_match,f1",
                "--max-samples",
                "3",
                "--output",
                tmpdir,
            ],
        )
        assert result.exit_code == 0
        assert "Exact Match" in result.stdout or "exact_match" in result.stdout


def test_cli_check() -> None:
    """Test 'urdu-eval check' health check command."""
    result = runner.invoke(app, ["check", "--provider", "mock"])
    assert result.exit_code == 0
    assert "Urdu Health Check" in result.stdout
    assert "Urdu comprehension" in result.stdout
    assert "Roman Urdu" in result.stdout


def test_cli_report_and_inspect() -> None:
    """Test 'urdu-eval report' and 'urdu-eval inspect' commands."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res = _create_mock_run_result()
        score_path = Path(tmpdir) / "scores.json"
        with score_path.open("w", encoding="utf-8") as f:
            f.write(res.model_dump_json())

        # Test report terminal
        rep_res = runner.invoke(app, ["report", str(score_path)])
        assert rep_res.exit_code == 0
        assert "exact_match" in rep_res.stdout

        # Test report HTML export
        html_out = Path(tmpdir) / "report.html"
        html_res = runner.invoke(app, ["report", str(score_path), "--html", str(html_out)])
        assert html_res.exit_code == 0
        assert html_out.exists()

        # Test inspect
        inspect_res = runner.invoke(app, ["inspect", str(score_path), "--limit", "2"])
        assert inspect_res.exit_code == 0
        assert "Prompt:" in inspect_res.stdout


def test_cli_leaderboard_and_compare() -> None:
    """Test 'urdu-eval leaderboard' and 'urdu-eval compare'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res = _create_mock_run_result()
        run_dir = Path(tmpdir) / res.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        score_path = run_dir / "scores.json"
        with score_path.open("w", encoding="utf-8") as f:
            f.write(res.model_dump_json())

        # Leaderboard table
        lb_res = runner.invoke(app, ["leaderboard", tmpdir])
        assert lb_res.exit_code == 0
        assert "mock-urdu-model" in lb_res.stdout

        # Leaderboard markdown
        lb_md = runner.invoke(app, ["leaderboard", tmpdir, "--format", "markdown"])
        assert lb_md.exit_code == 0
        assert "| Rank |" in lb_md.stdout

        # Compare
        comp_res = runner.invoke(app, ["compare", str(score_path)])
        assert comp_res.exit_code == 0
        assert "mock-urdu-model" in comp_res.stdout
