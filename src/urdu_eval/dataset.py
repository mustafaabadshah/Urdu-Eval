"""Dataset loading, streaming, and validation for UrduEval."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from urdu_eval.enums import Script
from urdu_eval.models import Sample

# Regex pattern for Arabic / Urdu Unicode blocks (0600-06FF, 0750-077F, FB50-FDFF, FE70-FEFF)
URDU_SCRIPT_REGEX = re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]")


class DatasetValidationResult(BaseModel):
    """Structured report produced by dataset validation."""

    is_valid: bool = True
    total_samples: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tasks_found: dict[str, int] = Field(default_factory=dict)
    scripts_found: dict[str, int] = Field(default_factory=dict)
    languages_found: dict[str, int] = Field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary of validation results."""
        if self.is_valid:
            lines = [
                f"Dataset is valid ({self.total_samples} samples verified).",
                f"  Tasks: {dict(self.tasks_found)}",
                f"  Scripts: {dict(self.scripts_found)}",
                f"  Languages: {dict(self.languages_found)}",
            ]
            if self.warnings:
                lines.append(f"  Warnings ({len(self.warnings)}):")
                lines.extend(f"    - {w}" for w in self.warnings[:10])
                if len(self.warnings) > 10:
                    lines.append(f"    ... and {len(self.warnings) - 10} more warnings")
            return "\n".join(lines)

        lines = [f"Dataset validation failed with {len(self.errors)} error(s):"]
        lines.extend(f"  ✗ {err}" for err in self.errors[:15])
        if len(self.errors) > 15:
            lines.append(f"  ... and {len(self.errors) - 15} more errors")
        return "\n".join(lines)


def compute_dataset_hash(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a dataset file for reproducibility."""
    hasher = hashlib.sha256()
    path = Path(file_path)
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def validate_dataset(file_path: str | Path) -> DatasetValidationResult:
    """Validate a JSONL dataset file against UrduEval schema and linguistic rules.

    Checks:
    - File existence & valid UTF-8 encoding
    - JSONL syntax per line
    - Unique sample IDs
    - Required fields (id, prompt, reference)
    - Valid task types, languages, and scripts
    - Empty prompt/reference checks
    - Script consistency warnings
    """
    path = Path(file_path)
    result = DatasetValidationResult()

    if not path.exists():
        result.is_valid = False
        result.errors.append(f"File not found: {path}")
        return result

    if not path.is_file():
        result.is_valid = False
        result.errors.append(f"Path is not a regular file: {path}")
        return result

    seen_ids: set[str] = set()

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue  # Skip blank lines

                try:
                    data: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as exc:
                    result.is_valid = False
                    result.errors.append(f"Line {line_no}: Invalid JSON syntax - {exc.msg}")
                    continue

                if not isinstance(data, dict):
                    result.is_valid = False
                    result.errors.append(
                        f"Line {line_no}: Expected JSON object, got {type(data).__name__}"
                    )
                    continue

                # Check ID uniqueness
                sample_id = data.get("id")
                if not sample_id:
                    result.is_valid = False
                    result.errors.append(f"Line {line_no}: Missing required field 'id'")
                elif str(sample_id) in seen_ids:
                    result.is_valid = False
                    result.errors.append(f"Line {line_no}: Duplicate sample ID '{sample_id}'")
                else:
                    seen_ids.add(str(sample_id))

                # Check reference
                ref = data.get("reference")
                if ref is None or (isinstance(ref, str) and not ref.strip()):
                    result.is_valid = False
                    result.errors.append(
                        f"Line {line_no} (ID: {sample_id or 'unknown'}): 'reference' cannot be empty or missing"
                    )
                elif isinstance(ref, list) and not ref:
                    result.is_valid = False
                    result.errors.append(
                        f"Line {line_no} (ID: {sample_id or 'unknown'}): Reference list cannot be empty"
                    )

                # Check prompt
                prompt = data.get("prompt")
                if not prompt or (isinstance(prompt, str) and not prompt.strip()):
                    result.is_valid = False
                    result.errors.append(
                        f"Line {line_no} (ID: {sample_id or 'unknown'}): 'prompt' cannot be empty or missing"
                    )

                # Validate with Pydantic Sample model
                try:
                    sample = Sample.model_validate(data)
                except ValidationError as exc:
                    result.is_valid = False
                    for err in exc.errors():
                        loc = " -> ".join(str(p) for p in err["loc"])
                        result.errors.append(
                            f"Line {line_no} (ID: {sample_id or 'unknown'}): Field '{loc}' - {err['msg']}"
                        )
                    continue

                # Collect statistics
                result.total_samples += 1
                task_str = str(sample.task.value)
                script_str = str(sample.script.value)
                lang_str = str(sample.language.value)

                result.tasks_found[task_str] = result.tasks_found.get(task_str, 0) + 1
                result.scripts_found[script_str] = result.scripts_found.get(script_str, 0) + 1
                result.languages_found[lang_str] = result.languages_found.get(lang_str, 0) + 1

                # Linguistic Script Consistency Warnings
                has_urdu_chars = bool(URDU_SCRIPT_REGEX.search(sample.prompt))
                if sample.script == Script.URDU and not has_urdu_chars:
                    result.warnings.append(
                        f"Line {line_no} (ID: {sample.id}): Script specified as 'urdu', but prompt contains no Perso-Arabic characters."
                    )
                elif sample.script == Script.ROMAN_URDU and has_urdu_chars:
                    result.warnings.append(
                        f"Line {line_no} (ID: {sample.id}): Script specified as 'roman_urdu', but prompt contains Perso-Arabic characters."
                    )

    except UnicodeDecodeError as exc:
        result.is_valid = False
        result.errors.append(f"Encoding error: File is not valid UTF-8 ({exc})")
        return result

    if result.total_samples == 0 and not result.errors:
        result.is_valid = False
        result.errors.append("Dataset is empty (no valid JSONL records found).")

    return result


def load_dataset(file_path: str | Path, max_samples: int | None = None) -> Iterator[Sample]:
    """Stream samples from a JSONL file.

    Does not load the entire dataset into memory at once.
    """
    path = Path(file_path)
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            sample = Sample.model_validate(data)
            yield sample
            count += 1
            if max_samples is not None and count >= max_samples:
                break
