"""Linguistic error and failure categorization."""

from __future__ import annotations

import re

from urdu_eval.enums import FailureCategory, Script, TaskType, TranslationDirection
from urdu_eval.models import Sample
from urdu_eval.normalization.roman_urdu import are_roman_urdu_variants
from urdu_eval.normalization.urdu import URDU_SCRIPT_REGEX

# Known refusal patterns in English and Urdu
REFUSAL_PATTERNS = [
    r"i cannot",
    r"i am unable",
    r"as an ai",
    r"i apologize",
    r"معذرت",
    r"معاف کیجیے",
    r"جواب دینے سے قاصر",
    r"پالیسی کے خلاف",
]
REFUSAL_REGEX = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def categorize_failure(
    sample: Sample,
    prediction: str,
    metrics: dict[str, float],
) -> FailureCategory:
    """Categorize prediction outcome into a standardized failure category.

    Does not modify scores; provides reproducible error taxonomy.
    """
    # 1. Check if prediction is correct according to primary metrics
    em = metrics.get("exact_match", 0.0)
    f1 = metrics.get("f1", 0.0)
    acc = metrics.get("accuracy", 0.0)

    if em >= 1.0 or acc >= 1.0 or f1 >= 0.99:
        return FailureCategory.CORRECT

    # 2. Check for refusal
    if REFUSAL_REGEX.search(prediction):
        return FailureCategory.REFUSAL

    # 3. Check for Script Violations
    has_urdu_chars = bool(URDU_SCRIPT_REGEX.search(prediction))
    has_latin_chars = bool(re.search(r"[a-zA-Z]", prediction))

    # Determine if Latin is expected (e.g. urdu_to_english translation or latin script)
    expects_latin = (
        sample.script == Script.LATIN or sample.direction == TranslationDirection.URDU_TO_ENGLISH
    )

    if sample.script == Script.URDU and not expects_latin:
        if has_latin_chars and not has_urdu_chars:
            return FailureCategory.WRONG_SCRIPT
    elif sample.script == Script.ROMAN_URDU:
        if has_urdu_chars:
            return FailureCategory.WRONG_SCRIPT

        # Check Roman Urdu spelling variation
        ref_texts = [sample.reference] if isinstance(sample.reference, str) else sample.reference
        pred_words = prediction.strip().split()
        if len(pred_words) == 1:
            for ref_item in ref_texts:
                if are_roman_urdu_variants(prediction.strip(), ref_item.strip()):
                    return FailureCategory.ROMAN_URDU_SPELLING

    # 4. Check for partial correctness
    if 0.2 <= f1 < 0.99:
        return FailureCategory.PARTIAL

    # 5. Task-specific failure categories
    if sample.task == TaskType.REASONING:
        return FailureCategory.REASONING_ERROR

    if sample.task == TaskType.TRANSLATION:
        return FailureCategory.TRANSLATION_DRIFT

    return FailureCategory.INCORRECT
