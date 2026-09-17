"""Tests for canonical cache key request hashing."""

from urdu_eval.runner.cache import compute_cache_key


def test_cache_key_deterministic() -> None:
    """Verify identical parameters produce identical SHA-256 cache keys."""
    k1 = compute_cache_key("ollama", "llama3.1", "پاکستان کہاں ہے؟", temperature=0.0, seed=42)
    k2 = compute_cache_key("Ollama", "llama3.1", "  پاکستان کہاں ہے؟  ", temperature=0.0, seed=42)
    assert k1 == k2


def test_cache_key_temperature_sensitivity() -> None:
    """Verify different temperatures produce distinct cache keys."""
    k1 = compute_cache_key("ollama", "llama3.1", "پاکستان", temperature=0.0)
    k2 = compute_cache_key("ollama", "llama3.1", "پاکستان", temperature=0.7)
    assert k1 != k2


def test_cache_key_seed_sensitivity() -> None:
    """Verify different seeds produce distinct cache keys."""
    k1 = compute_cache_key("ollama", "llama3.1", "پاکستان", seed=42)
    k2 = compute_cache_key("ollama", "llama3.1", "پاکستان", seed=100)
    assert k1 != k2


def test_cache_key_prompt_template_version_sensitivity() -> None:
    """Verify distinct prompt protocol versions yield distinct cache keys."""
    k1 = compute_cache_key("ollama", "llama3.1", "پاکستان", prompt_template_version="1.0")
    k2 = compute_cache_key("ollama", "llama3.1", "پاکستان", prompt_template_version="2.0")
    assert k1 != k2
