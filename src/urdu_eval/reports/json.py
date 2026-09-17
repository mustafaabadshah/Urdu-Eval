"""JSON evaluation report persistence."""

from __future__ import annotations

from pathlib import Path

from urdu_eval.models import RunResult


def export_json_report(run_result: RunResult, file_path: str | Path) -> None:
    """Serialize RunResult to formatted JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(run_result.model_dump_json(indent=2))
