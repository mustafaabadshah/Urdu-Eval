"""Deterministic Multiple Choice Question (MCQ) extraction protocol for Urdu evaluation."""

from __future__ import annotations

import re

# Standard Latin choice letters
VALID_CHOICES = ("A", "B", "C", "D")

# Regex patterns for identifying choice letters with common Urdu/English prefixes
_CHOICE_PATTERNS = [
    # Explicit Urdu prefix: درست جواب C ہے / جواب: C / صحیح جواب (A)
    re.compile(
        r"(?:درست|صحیح|صحیح ترین)\s+(?:جواب|انتخاب|آپشن|حرف)\s*[:\-]?\s*[\(\[\{]?([A-Da-d])[\)\]\}]?",
        re.UNICODE,
    ),
    re.compile(r"(?:جواب|انتخاب|آپشن)\s*[:\-]?\s*[\(\[\{]?([A-Da-d])[\)\]\}]?", re.UNICODE),
    # English prefix: Answer: C / Choice: (B) / The correct option is A
    re.compile(
        r"(?:answer|choice|option|correct option is)\s*[:\-]?\s*[\(\[\{]?([A-Da-d])[\)\]\}]?",
        re.IGNORECASE,
    ),
    # Urdu postfix: C درست ہے / B صحیح ہے
    re.compile(r"[\(\[\{]?([A-Da-d])[\)\]\}]?\s*(?:درست|صحیح)\s*(?:ہے|ہوگا)", re.UNICODE),
    # Standalone letter with delimiters: (A), [B], C., **D**
    re.compile(r"^\s*[\(\[\{]?\**([A-Da-d])\**[\)\]\}]?[\.\:\-\s]*$"),
]


def extract_mcq_choice(
    response_text: str,
    valid_choices: tuple[str, ...] | list[str] = VALID_CHOICES,
    options: list[str] | None = None,
) -> str | None:
    """Extract standard choice letter (A, B, C, D) deterministically from model response.

    Protocol v1:
    1. Direct match on isolated choice letters or letters with standard Urdu/English answer prefixes.
    2. Regex scan for bounded letters with explicit delimiters.
    3. If choice letter not found, match against full option texts if options are provided.
    4. First prominent valid choice letter in the response.

    Returns uppercase choice letter (e.g. 'C') or None if no valid choice can be determined.
    """
    if not response_text:
        return None

    cleaned = response_text.strip()
    valid_set = {c.upper() for c in valid_choices}

    # 1. Check exact match: single letter
    if cleaned.upper() in valid_set:
        return cleaned.upper()

    # 2. Check prioritized regex patterns
    for pat in _CHOICE_PATTERNS:
        match = pat.search(cleaned)
        if match:
            candidate = match.group(1).upper()
            if candidate in valid_set:
                return candidate

    # 3. Fallback: match by option text if options are provided
    if options:
        cleaned_lower = cleaned.lower()
        for idx, opt in enumerate(options):
            if not opt:
                continue
            opt_clean = opt.strip().lower()
            if len(opt_clean) > 2 and opt_clean in cleaned_lower:
                if idx < len(valid_choices):
                    return valid_choices[idx].upper()

    # 4. Final heuristic: find any isolated letter A-D surrounded by word boundaries or punctuation
    isolated_match = re.search(
        r"(?:^|[\s\(\[\{\:\.\,\-])([A-Da-d])(?:[\s\)\]\}\:\.\,\-]|$)", cleaned
    )
    if isolated_match:
        cand = isolated_match.group(1).upper()
        if cand in valid_set:
            return cand

    return None
