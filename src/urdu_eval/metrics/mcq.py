"""Deterministic Multiple Choice Question (MCQ) extraction protocol for Urdu evaluation."""

from __future__ import annotations

import re

# Standard Latin choice letters
VALID_CHOICES = ("A", "B", "C", "D")

# Regex patterns for identifying choice letters with common Urdu/English prefixes
_CHOICE_PATTERNS = [
    # Explicit Urdu prefix: درست جواب C ہے / جواب: C / صحیح ترین آپشن: (A)
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

# Ambiguity and hedging patterns between candidate choices
_DISJUNCTION_CONJUNCTION_PATTERN = re.compile(
    r"(?:\b|[\(\[\{\s])([A-Da-d])\s*(?:یا|اور|یا\s*پھر|یا\s*شاید|لیکن|بلکہ|\/|\\|\||or|and|either|neither)\s*(?:شاید\s*)?[\(\[\{\s]*([A-Da-d])(?:\b|[\)\]\}\s])",
    re.UNICODE | re.IGNORECASE,
)

_HEDGING_MULTI_PATTERN = re.compile(
    r"(?:لگتا|ممکن|شاید|ہو سکتا|doubt|think|maybe|perhaps)\s*.*?\b([A-Da-d])\b.*?(?:شاید|لیکن|یا|بلکہ|اگرچہ|حالانکہ|or|but|maybe)\s*.*?\b([A-Da-d])\b",
    re.UNICODE | re.IGNORECASE,
)

_HEDGING_QUALIFICATION = re.compile(
    r"\b([A-Da-d])\b.*?(?:اگرچہ|حالانکہ|لیکن|مگر|although|though|however)\s*.*?\b([A-Da-d])\b.*?(?:ممکن|شاید|ہو سکتا|possible|plausible)",
    re.UNICODE | re.IGNORECASE,
)


def extract_mcq_choice(
    response_text: str,
    valid_choices: tuple[str, ...] | list[str] = VALID_CHOICES,
    options: list[str] | None = None,
) -> str | None:
    """Extract standard choice letter (A, B, C, D) deterministically from model response.

    Refuses ambiguous, conflicting, or hedged responses (returns None) to eliminate false positives:
    - Rejects disjunctive/conjunctive choices (e.g., 'A یا C', 'A اور B', 'A or C').
    - Rejects hedged choices (e.g., 'مجھے لگتا ہے A، لیکن شاید C').
    - Rejects responses enumerating multiple choices without a singular explicit answer declaration (e.g., 'A) ... B) ...').
    - If options are provided and multiple distinct option texts appear, refuses extraction.

    Returns uppercase choice letter (e.g. 'C') or None if ambiguous or undetermined.
    """
    if not response_text:
        return None

    cleaned = response_text.strip()
    valid_set = {c.upper() for c in valid_choices}

    # 1. Exact match on single token
    if cleaned.upper() in valid_set:
        return cleaned.upper()

    # 2. Check for explicit disjunction / conjunction between conflicting choices
    disj_match = _DISJUNCTION_CONJUNCTION_PATTERN.search(cleaned)
    if disj_match:
        c1, c2 = disj_match.group(1).upper(), disj_match.group(2).upper()
        if c1 in valid_set and c2 in valid_set and c1 != c2:
            return None  # Ambiguous: e.g. "A یا C", "A اور B", "A or C"

    hedg_match = _HEDGING_MULTI_PATTERN.search(cleaned)
    if hedg_match:
        c1, c2 = hedg_match.group(1).upper(), hedg_match.group(2).upper()
        if c1 in valid_set and c2 in valid_set and c1 != c2:
            return None  # Ambiguous: e.g. "مجھے لگتا ہے A، لیکن شاید C"

    hedg_qual = _HEDGING_QUALIFICATION.search(cleaned)
    if hedg_qual:
        c1, c2 = hedg_qual.group(1).upper(), hedg_qual.group(2).upper()
        if c1 in valid_set and c2 in valid_set and c1 != c2:
            return None  # Ambiguous: e.g. "میرا حتمی جواب C ہے، اگرچہ A بھی ممکن ہے"

    # 3. Check prioritized explicit declaration patterns
    declared_candidates: list[str] = []
    for pat in _CHOICE_PATTERNS:
        for match in pat.finditer(cleaned):
            cand = match.group(1).upper()
            if cand in valid_set and cand not in declared_candidates:
                declared_candidates.append(cand)

    if len(declared_candidates) == 1:
        return declared_candidates[0]
    if len(declared_candidates) > 1:
        return None  # Multiple conflicting explicit declarations (e.g. "درست جواب A ہے ... جواب B")

    # 4. Fallback: match by option text if options are provided
    if options:
        cleaned_lower = cleaned.lower()
        matched_option_indices: list[int] = []
        for idx, opt in enumerate(options):
            if not opt:
                continue
            opt_clean = opt.strip().lower()
            if len(opt_clean) > 2 and opt_clean in cleaned_lower:
                if idx < len(valid_choices):
                    matched_option_indices.append(idx)

        # Only accept if EXACTLY ONE option text matches
        if len(matched_option_indices) == 1:
            return valid_choices[matched_option_indices[0]].upper()
        if len(matched_option_indices) > 1:
            return None  # Multiple option texts mentioned: ambiguous

    # 5. Scan for isolated candidate choice letters
    isolated_letters = re.findall(
        r"(?:^|[\s\(\[\{\:\.\,\-])([A-Da-d])(?:[\s\)\]\}\:\.\,\-]|$)", cleaned
    )
    isolated_valid = [c.upper() for c in isolated_letters if c.upper() in valid_set]
    unique_isolated = set(isolated_valid)

    # If exactly ONE distinct choice letter is mentioned throughout the entire text, accept it
    if len(unique_isolated) == 1:
        return str(list(unique_isolated)[0])

    # If multiple distinct choice letters are present (e.g. "A) ... B) ..."), reject as ambiguous!
    return None
