"""Normalization modules for Urdu and Roman Urdu text."""

from urdu_eval.normalization.roman_urdu import (
    are_roman_urdu_variants,
    normalize_roman_urdu,
)
from urdu_eval.normalization.text import (
    normalize_punctuation,
    normalize_whitespace,
)
from urdu_eval.normalization.urdu import (
    normalize_arabic_variants,
    normalize_digits,
    normalize_urdu,
    remove_diacritics,
)

__all__ = [
    "normalize_whitespace",
    "normalize_punctuation",
    "remove_diacritics",
    "normalize_arabic_variants",
    "normalize_digits",
    "normalize_urdu",
    "normalize_roman_urdu",
    "are_roman_urdu_variants",
]
