"""Tests for leaderboard aggregation and export."""

import tempfile
from pathlib import Path

from tests.test_reports import _create_mock_run_result
from urdu_eval.leaderboard import (
    build_leaderboard,
    export_leaderboard_csv,
    export_leaderboard_json,
    export_leaderboard_markdown,
)


def test_leaderboard_builder_and_exports() -> None:
    """Test building leaderboard from directory of runs and exporting to formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_res = _create_mock_run_result()
        run_dir = Path(tmpdir) / run_res.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        scores_file = run_dir / "scores.json"
        with scores_file.open("w", encoding="utf-8") as f:
            f.write(run_res.model_dump_json())

        # Build leaderboard
        entries = build_leaderboard(tmpdir)
        assert len(entries) == 1
        assert entries[0].model == "mock-urdu-model"
        assert entries[0].benchmark_id == "urdu-qa"

        # Test Markdown export
        md = export_leaderboard_markdown(entries)
        assert "| Rank | Model |" in md
        assert "mock-urdu-model" in md

        # Test CSV export
        csv_str = export_leaderboard_csv(entries)
        assert "rank,model,provider" in csv_str
        assert "mock-urdu-model" in csv_str

        # Test JSON export
        json_str = export_leaderboard_json(entries)
        assert "mock-urdu-model" in json_str
