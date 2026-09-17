"""Response caching to prevent duplicate API costs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from urdu_eval.models import ModelResponse


def compute_cache_key(
    provider: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    top_p: float | None = 1.0,
    seed: int | None = 42,
    system_prompt: str | None = None,
    benchmark_id: str | None = None,
    benchmark_version: str | None = None,
    prompt_template_version: str | None = None,
    extra_params: dict[str, Any] | None = None,
) -> str:
    """Generate a canonical SHA-256 cache key from all request parameters that affect generation.

    Guarantees:
    - Any alteration in hyperparameters (temperature, top_p, seed) produces a distinct key.
    - Distinct system prompts, template versions, or benchmark versions will never collide in cache.
    """
    payload = {
        "provider": provider.lower().strip(),
        "model": model.lower().strip(),
        "prompt": prompt.strip(),
        "temperature": round(temperature, 4),
        "max_tokens": max_tokens,
        "top_p": round(top_p, 4) if top_p is not None else 1.0,
        "seed": seed,
        "system_prompt": (system_prompt or "").strip(),
        "benchmark_id": (benchmark_id or "").lower().strip(),
        "benchmark_version": (benchmark_version or "").strip(),
        "prompt_template_version": (prompt_template_version or "1.0").strip(),
        "extra_params": extra_params or {},
    }
    canonical_request = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()


class EvaluationCache:
    """SQLite-backed persistent response cache."""

    def __init__(self, cache_dir: str | Path = ".urdu_eval_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "responses.db"
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager that ensures the SQLite connection is always closed cleanly."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT,
                    model TEXT,
                    response_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get(self, cache_key: str) -> ModelResponse | None:
        """Retrieve cached ModelResponse if present."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response_json FROM response_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return ModelResponse.model_validate(data)
        return None

    def set(self, cache_key: str, response: ModelResponse) -> None:
        """Store ModelResponse in cache."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO response_cache (cache_key, provider, model, response_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    cache_key,
                    response.provider,
                    response.model,
                    json.dumps(response.model_dump(mode="json")),
                ),
            )
            conn.commit()

    def clear(self) -> None:
        """Clear all cached responses."""
        with self._connect() as conn:
            conn.execute("DELETE FROM response_cache")
            conn.commit()

    def count(self) -> int:
        """Return total number of cached responses."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM response_cache")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def get_stats(self) -> dict[str, Any]:
        """Return summary cache statistics."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*), COUNT(DISTINCT provider), COUNT(DISTINCT model) FROM response_cache"
            )
            row = cursor.fetchone()
            total, providers, models = row if row else (0, 0, 0)
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            return {
                "total_entries": total,
                "providers": providers,
                "models": models,
                "db_size_bytes": db_size,
                "db_path": str(self.db_path),
            }
