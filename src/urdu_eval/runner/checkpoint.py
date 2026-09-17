"""Checkpoint and resume manager for long-running evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from urdu_eval.models import SampleResult


class CheckpointManager:
    """Manages appending and reloading sample-level results for resilient resumption."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.run_dir / "checkpoint.jsonl"
        self._completed_ids: set[str] = set()
        self._completed_results: list[SampleResult] = []
        self._load_existing_checkpoint()

    def _load_existing_checkpoint(self) -> None:
        if not self.checkpoint_file.exists():
            return

        with self.checkpoint_file.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data: dict[str, Any] = json.loads(stripped)
                    sample_res = SampleResult.model_validate(data)
                    self._completed_ids.add(sample_res.sample_id)
                    self._completed_results.append(sample_res)
                except Exception:
                    continue

    def is_completed(self, sample_id: str) -> bool:
        """Check if a sample has already been evaluated in this run."""
        return sample_id in self._completed_ids

    def record_sample(self, result: SampleResult) -> None:
        """Atomically append a completed sample result to the checkpoint log."""
        self._completed_ids.add(result.sample_id)
        self._completed_results.append(result)

        with self.checkpoint_file.open("a", encoding="utf-8") as f:
            line = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
            f.write(f"{line}\n")

    def get_completed_results(self) -> list[SampleResult]:
        """Return all completed sample results."""
        return list(self._completed_results)

    @property
    def completed_count(self) -> int:
        return len(self._completed_ids)
