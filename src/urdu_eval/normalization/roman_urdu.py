"""Conservative Roman Urdu normalization and orthographic variation handlers."""

from __future__ import annotations

import re

from urdu_eval.normalization.text import normalize_whitespace

# Regex for repeated vowels (3 or more consecutive identical vowels: "shukriyaaa" -> "shukriya")
REPEATED_VOWELS_REGEX = re.compile(r"([aeiou])\1{2,}", re.IGNORECASE)

# Punctuation to strip in Roman Urdu
ROMAN_PUNCT_REGEX = re.compile(r"[\.,!?:;\-\–\—\(\)\[\]\{\}\"\'`~@#$%^&*+=/<>\\]")

# Known common phonetic / orthographic variation clusters in Roman Urdu (for diagnostic comparison)
COMMON_ROMAN_URDU_VARIATION_CLUSTERS = [
    {"shukria", "shukriya", "shukriyah", "shukriyaa"},
    {"kya", "kia", "kyaa"},
    {"hai", "hy", "hye"},
    {"hain", "hn", "heen"},
    {"mujhe", "mjhe", "mujhey", "mujy"},
    {"tujhe", "tjhe", "tujhey", "tujy"},
    {"lekin", "lkn", "laikin"},
    {"kyun", "kyu", "kyon", "kion"},
    {"bhi", "b", "bh"},
    {"kaise", "kese", "kaisay"},
    {"aise", "ese", "aisay"},
    {"bohat", "bht", "bahut", "bohot", "boht"},
    {"acha", "achha", "achaa"},
    {"nahi", "nahin", "nhi", "nahee"},
    {"karachi", "karaci"},
    {"islamabad", "islam abad"},
    {"khubsurat", "khoobsurat", "khubsoorat", "khoobsoorat"},
    {"zaroorat", "zarurat"},
]


def normalize_roman_urdu(
    text: str,
    lower: bool = True,
    remove_punct: bool = True,
    collapse_vowels: bool = False,
) -> str:
    """Normalize Roman Urdu text conservatively.

    Parameters:
    - text: raw Roman Urdu string
    - lower: whether to lowercase
    - remove_punct: whether to remove punctuation
    - collapse_vowels: whether to collapse 3+ repeated vowels (e.g. 'booohat' -> 'bohat')
    """
    if not text:
        return ""

    cleaned = text
    if lower:
        cleaned = cleaned.lower()

    if remove_punct:
        cleaned = ROMAN_PUNCT_REGEX.sub(" ", cleaned)

    if collapse_vowels:
        cleaned = REPEATED_VOWELS_REGEX.sub(r"\1", cleaned)

    return normalize_whitespace(cleaned)


def normalize_roman_urdu_strict(text: str) -> str:
    """Strict Roman Urdu normalization (casing, whitespace, punctuation only; zero vowel mutation)."""
    return normalize_roman_urdu(text, lower=True, remove_punct=True, collapse_vowels=False)


def normalize_roman_urdu_phonetic(text: str) -> str:
    """Phonetic Roman Urdu normalization with safe elongation collapse (3+ identical vowels collapsed)."""
    return normalize_roman_urdu(text, lower=True, remove_punct=True, collapse_vowels=True)


def are_roman_urdu_variants(word1: str, word2: str) -> bool:
    """Check whether two words belong to known Roman Urdu orthographic variation clusters.

    Does not modify text; provides a diagnostic signal.
    """
    w1 = normalize_roman_urdu(word1, lower=True, remove_punct=True)
    w2 = normalize_roman_urdu(word2, lower=True, remove_punct=True)

    if w1 == w2:
        return True

    for cluster in COMMON_ROMAN_URDU_VARIATION_CLUSTERS:
        if w1 in cluster and w2 in cluster:
            return True

    return False
