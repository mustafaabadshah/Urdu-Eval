"""Conservative Urdu script normalization and diacritic removal."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from urdu_eval.normalization.text import normalize_whitespace

# Regex pattern for Arabic / Urdu Unicode blocks (0600-06FF, 0750-077F, FB50-FDFF, FE70-FEFF)
URDU_SCRIPT_REGEX = re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]")

# Urdu aerab / diacritics:
# Zabar (064E), Zer (0650), Pesh (064F), Tashdeed (0651), Jazm/Sukun (0652),
# Tanween Fatha (064B), Tanween Damma (064C), Tanween Kasra (064D),
# Khada Zabar (0670), Khada Zer (0656), Ulta Pesh (0657)
DIACRITICS_REGEX = re.compile(r"[\u064B-\u0657\u0670\u06DF-\u06E4\u06EA-\u06ED]")

# Conservative Arabic-to-Urdu character harmonization map:
# Only uncontroversial keyboard orthographic variants:
# Arabic Kaf (ك U+0643) -> Urdu Keheh (ک U+06A9)
# Arabic Yeh (ي U+064A) / Alif Maksura (ى U+0649) -> Urdu Choti Yeh (ی U+06CC)
# NOTE: Do-Chashmi Heh (ھ U+06BE) is strictly preserved for aspirated consonants (بھ, پھ, تھ, etc.).
# Teh Marbuta (ة U+0629) is PRESERVED in conservative profile to prevent destructive lexical merging.
CONSERVATIVE_CHAR_MAP = {
    "\u0643": "\u06a9",  # ك -> ک
    "\u064a": "\u06cc",  # ي -> ی
    "\u0649": "\u06cc",  # ى -> ی
}

# Standard harmonization map (includes Arabic Heh)
STANDARD_CHAR_MAP = {
    **CONSERVATIVE_CHAR_MAP,
    "\u0647": "\u06c1",  # ه -> ہ (Arabic Heh to Urdu Gol Heh)
}

# Legacy alias for backward compatibility
ARABIC_TO_URDU_CHAR_MAP = CONSERVATIVE_CHAR_MAP

# Urdu numerals (۰-۹) to standard Arabic numerals (0-9)
URDU_NUMERALS_MAP = {
    "۰": "0",
    "۱": "1",
    "۲": "2",
    "۳": "3",
    "۴": "4",
    "۵": "5",
    "۶": "6",
    "۷": "7",
    "۸": "8",
    "۹": "9",
    "٠": "0",
    "١": "1",
    "٢": "2",
    "٣": "3",
    "٤": "4",
    "٥": "5",
    "٦": "6",
    "٧": "7",
    "٨": "8",
    "٩": "9",
}


def remove_diacritics(text: str) -> str:
    """Remove Urdu/Arabic vowel diacritics (aerab/harakaat) while preserving consonants."""
    return DIACRITICS_REGEX.sub("", text)


def normalize_arabic_variants(text: str, conservative_only: bool = True) -> str:
    """Harmonize Arabic keyboard character variants into standard Urdu characters.

    In conservative mode (default), converts only Kaf and Yeh.
    Preserves Teh Marbuta (ة) and Do-Chashmi Heh (ھ) to avoid linguistic loss.
    """
    mapping = CONSERVATIVE_CHAR_MAP if conservative_only else STANDARD_CHAR_MAP
    cleaned = text
    for ar_char, ur_char in mapping.items():
        cleaned = cleaned.replace(ar_char, ur_char)
    return cleaned


def normalize_digits(text: str) -> str:
    """Convert Eastern Arabic and Urdu numerals (۰-۹) to ASCII digits (0-9)."""
    cleaned = text
    for ur_num, en_num in URDU_NUMERALS_MAP.items():
        cleaned = cleaned.replace(ur_num, en_num)
    return cleaned


def normalize_urdu(
    text: str,
    remove_aerab: bool = True,
    harmonize_chars: bool = True,
    convert_digits: bool = False,
    conservative_only: bool = True,
    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC",
) -> str:
    """Perform conservative Urdu normalization.

    Parameters:
    - text: input Urdu string
    - remove_aerab: whether to remove diacritic vowels (zabar, zer, pesh, etc.)
    - harmonize_chars: whether to convert Arabic Kaf/Yeh to standard Urdu Unicode
    - convert_digits: whether to convert Urdu numerals (۰-۹) to 0-9
    - conservative_only: preserve Teh Marbuta and Arabic Heh distinctions
    - unicode_form: Unicode normalization form ('NFC', 'NFD', etc.)
    """
    if not text:
        return ""

    normalized = unicodedata.normalize(unicode_form, text)

    if harmonize_chars:
        normalized = normalize_arabic_variants(normalized, conservative_only=conservative_only)

    if remove_aerab:
        normalized = remove_diacritics(normalized)

    if convert_digits:
        normalized = normalize_digits(normalized)

    return normalize_whitespace(normalized)


def normalize_with_profile(text: str, profile: str = "conservative") -> str:
    """Normalize text according to a named NormalizationProfile.

    Profiles:
    - 'raw': Trims surrounding whitespace without modifying any characters.
    - 'conservative' (default): Safe NFC Unicode normalization, Kaf/Yeh harmonization,
      aerab stripping, and whitespace collapse. Preserves digits, Teh Marbuta, and Heh.
    - 'standard': Conservative + Arabic Heh harmonization, Eastern Arabic digit conversion,
      and punctuation harmonization.
    - 'roman_urdu': Latin-script normalization (lowercasing, vowel elongation collapse, punctuation).
    """
    if not text:
        return ""

    prof = profile.lower() if isinstance(profile, str) else profile.value.lower()

    if prof == "raw":
        return text.strip()

    if prof == "conservative":
        return normalize_urdu(
            text,
            remove_aerab=True,
            harmonize_chars=True,
            convert_digits=False,
            conservative_only=True,
        )

    if prof == "standard" or prof == "urdueval":
        return normalize_urdu(
            text,
            remove_aerab=True,
            harmonize_chars=True,
            convert_digits=True,
            conservative_only=False,
        )

    if prof == "roman_urdu":
        from urdu_eval.normalization.roman_urdu import normalize_roman_urdu

        return normalize_roman_urdu(text)

    # Fallback to conservative
    return normalize_urdu(
        text, remove_aerab=True, harmonize_chars=True, convert_digits=False, conservative_only=True
    )
