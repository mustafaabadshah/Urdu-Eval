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
    """Categorize prediction outcome into a principled, task-aware error taxonomy.

    Uses task- and metric-specific thresholds rather than arbitrary universal cutoffs.
    Does not modify scores; provides reproducible diagnostic classifications.
    """
    clean_pred = prediction.strip()
    em = metrics.get("exact_match", 0.0)
    acc = metrics.get("accuracy", 0.0)
    f1 = metrics.get("f1", 0.0)
    chrf = metrics.get("chrf", metrics.get("chrf++", 0.0))
    bleu = metrics.get("bleu", 0.0)
    rouge_l = metrics.get("rouge-l", metrics.get("rouge_l", 0.0))
    judge = metrics.get("judge", 0.0)

    # 1. Check for Model Refusal across English and Urdu refusal signals
    if REFUSAL_REGEX.search(clean_pred):
        return FailureCategory.REFUSAL

    # 2. Check for Script Compliance & Language Drift
    has_urdu_chars = bool(URDU_SCRIPT_REGEX.search(clean_pred))
    has_latin_chars = bool(re.search(r"[a-zA-Z]", clean_pred))

    expects_latin = (
        sample.script == Script.LATIN
        or sample.direction == TranslationDirection.URDU_TO_ENGLISH
        or (
            sample.task == TaskType.TRANSLATION
            and sample.direction == TranslationDirection.URDU_TO_ENGLISH
        )
    )

    if sample.script == Script.URDU and not expects_latin:
        if has_latin_chars and not has_urdu_chars:
            return FailureCategory.WRONG_SCRIPT
    elif sample.script == Script.ROMAN_URDU:
        if has_urdu_chars:
            return FailureCategory.WRONG_SCRIPT

        # Check for Roman Urdu spelling variation
        ref_texts = [sample.reference] if isinstance(sample.reference, str) else sample.reference
        pred_words = clean_pred.split()
        if len(pred_words) == 1:
            for ref_item in ref_texts:
                if are_roman_urdu_variants(clean_pred, str(ref_item).strip()):
                    return FailureCategory.ROMAN_URDU_SPELLING

    # 3. Task-Specific Diagnostic Thresholds
    # A. Question Answering / MMLU
    if sample.task in (TaskType.QA, TaskType.MMLU):
        if em >= 1.0 or acc >= 1.0:
            return FailureCategory.CORRECT
        if f1 >= 0.75:
            return FailureCategory.CORRECT
        if 0.25 <= f1 < 0.75:
            return FailureCategory.PARTIAL
        return FailureCategory.INCORRECT

    # B. Translation
    if sample.task == TaskType.TRANSLATION:
        # Check translation drift (echoing prompt language instead of translating)
        if (
            sample.direction == TranslationDirection.URDU_TO_ENGLISH
            and has_urdu_chars
            and not has_latin_chars
        ):
            return FailureCategory.TRANSLATION_DRIFT
        if (
            sample.direction == TranslationDirection.ENGLISH_TO_URDU
            and has_latin_chars
            and not has_urdu_chars
        ):
            return FailureCategory.TRANSLATION_DRIFT

        # Metric thresholds for translation
        if chrf >= 0.65 or bleu >= 0.50:
            return FailureCategory.CORRECT
        if chrf >= 0.35 or bleu >= 0.20:
            return FailureCategory.PARTIAL
        return (
            FailureCategory.TRANSLATION_DRIFT
            if (chrf < 0.20 and bleu < 0.10)
            else FailureCategory.INCORRECT
        )

    # C. Multi-Step Reasoning
    if sample.task == TaskType.REASONING:
        if em >= 1.0 or acc >= 1.0:
            return FailureCategory.CORRECT
        if f1 >= 0.80:
            return FailureCategory.CORRECT
        return FailureCategory.REASONING_ERROR

    # D. Summarization
    if sample.task == TaskType.SUMMARIZATION:
        if rouge_l >= 0.55 or f1 >= 0.60:
            return FailureCategory.CORRECT
        if 0.25 <= rouge_l < 0.55:
            return FailureCategory.PARTIAL
        return FailureCategory.INCORRECT

    # E. General Fallback
    if em >= 1.0 or acc >= 1.0 or judge >= 0.80 or f1 >= 0.80:
        return FailureCategory.CORRECT
    if f1 >= 0.30 or judge >= 0.40:
        return FailureCategory.PARTIAL

    return FailureCategory.INCORRECT
