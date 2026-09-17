"""Generic whitespace and punctuation normalization."""

from __future__ import annotations

import re

# Whitespace regex covering standard and unicode spaces
WHITESPACE_REGEX = re.compile(r"[\s\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]+")

# Urdu and general punctuation mappings
URDU_TO_STANDARD_PUNCT = {
    "\u06d4": ".",  # Urdu full stop (۔)
    "\u060c": ",",  # Urdu comma (،)
    "\u061b": ";",  # Urdu semicolon (؛)
    "\u061f": "?",  # Urdu question mark (؟)
    "\u2018": "'",  # Left single quote
    "\u2019": "'",  # Right single quote
    "\u201c": '"',  # Left double quote
    "\u201d": '"',  # Right double quote
}


def normalize_whitespace(text: str) -> str:
    """Trim leading/trailing whitespace and collapse internal whitespace runs to a single space."""
    return WHITESPACE_REGEX.sub(" ", text).strip()


def normalize_punctuation(text: str, replace_urdu_punct: bool = False) -> str:
    """Normalize punctuation in text.

    If replace_urdu_punct is True, converts Urdu-specific punctuation (۔ ، ؛ ؟)
    to Latin counterparts (. , ; ?).
    """
    cleaned = text
    if replace_urdu_punct:
        for urdu_p, std_p in URDU_TO_STANDARD_PUNCT.items():
            cleaned = cleaned.replace(urdu_p, std_p)
    return normalize_whitespace(cleaned)
