"""Tests for dataset loading, streaming, and validation."""

import tempfile
from pathlib import Path

from urdu_eval.dataset import compute_dataset_hash, load_dataset, validate_dataset
from urdu_eval.enums import Script


def test_validate_valid_dataset() -> None:
    """Validate the included examples/sample_qa.jsonl."""
    sample_file = Path("examples/sample_qa.jsonl")
    assert sample_file.exists()

    result = validate_dataset(sample_file)
    assert result.is_valid is True
    assert result.total_samples == 5
    assert len(result.errors) == 0
    assert result.tasks_found.get("qa") == 3
    assert result.tasks_found.get("translation") == 1
    assert result.tasks_found.get("reasoning") == 1
    assert result.scripts_found.get("urdu") == 3
    assert result.scripts_found.get("roman_urdu") == 2

    # Check hash is valid 64-character SHA-256 hex digest
    h = compute_dataset_hash(sample_file)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_validate_invalid_dataset() -> None:
    """Validate that invalid dataset catches duplicate IDs, JSON errors, missing fields."""
    invalid_file = Path("examples/invalid_sample.jsonl")
    assert invalid_file.exists()

    result = validate_dataset(invalid_file)
    assert result.is_valid is False
    assert len(result.errors) > 0

    error_str = " ".join(result.errors)
    assert "Duplicate sample ID" in error_str
    assert "Invalid JSON syntax" in error_str
    assert "cannot be empty" in error_str


def test_validate_nonexistent_file() -> None:
    """Validate behavior on missing file."""
    result = validate_dataset("nonexistent_path.jsonl")
    assert result.is_valid is False
    assert any("File not found" in err for err in result.errors)


def test_validate_script_warning() -> None:
    """Verify script heuristic warnings when script and text mismatch."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as f:
        # Script is 'urdu', but prompt is pure English/ASCII
        f.write(
            '{"id": "warn_01", "task": "qa", "script": "urdu", "prompt": "Where is Lahore?", "reference": "Pakistan"}\n'
        )
        temp_path = f.name

    try:
        res = validate_dataset(temp_path)
        assert res.is_valid is True
        assert len(res.warnings) == 1
        assert "contains no Perso-Arabic characters" in res.warnings[0]
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_load_dataset_streaming() -> None:
    """Test streaming generator for dataset."""
    sample_file = Path("examples/sample_qa.jsonl")
    samples = list(load_dataset(sample_file))
    assert len(samples) == 5
    assert samples[0].id == "pk_001"
    assert samples[0].script == Script.URDU
    assert samples[1].script == Script.ROMAN_URDU

    # Test max_samples limit
    partial = list(load_dataset(sample_file, max_samples=2))
    assert len(partial) == 2
    assert partial[0].id == "pk_001"
    assert partial[1].id == "pk_002"
