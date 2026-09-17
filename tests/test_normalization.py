"""Tests for Urdu and Roman Urdu text normalization."""

from urdu_eval.normalization import (
    are_roman_urdu_variants,
    normalize_arabic_variants,
    normalize_digits,
    normalize_punctuation,
    normalize_roman_urdu,
    normalize_urdu,
    normalize_whitespace,
    remove_diacritics,
)


def test_normalize_whitespace() -> None:
    """Test collapsing multiple spaces, tabs, and unicode spaces."""
    raw = "   پاکستان    ایک    ملک    ہے   \n  "
    assert normalize_whitespace(raw) == "پاکستان ایک ملک ہے"


def test_remove_diacritics() -> None:
    """Test removing aerab (zabar, zer, pesh, jazm, tashdeed)."""
    # پَاکِسْتَانْ (with diacritics)
    with_aerab = "\u067e\u064e\u0627\u06a9\u0650\u0633\u0652\u062a\u064e\u0627\u0646\u0652"
    clean = remove_diacritics(with_aerab)
    assert clean == "پاکستان"


def test_normalize_arabic_variants() -> None:
    """Test converting Arabic Kaf (ك) and Yeh (ي) to standard Urdu Unicode."""
    # Arabic keyboard spelling: پاكستان (Arabic Kaf 0643)
    arabic_kaf = "پا\u0643ستان"
    assert normalize_arabic_variants(arabic_kaf) == "پاکستان"

    # Arabic Yeh (ي 064A)
    arabic_yeh = "عل\u064a"
    assert normalize_arabic_variants(arabic_yeh) == "علی"


def test_normalize_digits() -> None:
    """Test converting Urdu numerals to standard digits."""
    urdu_nums = "۱۲۳۴۵"
    assert normalize_digits(urdu_nums) == "12345"


def test_normalize_urdu_complete() -> None:
    """Test full Urdu normalization pipeline."""
    raw = "  پاكستان  ایک  خُوبصُورتْ  مُلک  ہے  "
    normalized = normalize_urdu(raw, remove_aerab=True, harmonize_chars=True)
    assert normalized == "پاکستان ایک خوبصورت ملک ہے"


def test_normalize_punctuation() -> None:
    """Test Urdu punctuation normalization."""
    raw = "پاکستان، ایک عظیم ملک ہے؛ کیا آپ جانتے ہیں؟ جی ہاں۔"
    converted = normalize_punctuation(raw, replace_urdu_punct=True)
    assert "پاکستان, ایک عظیم ملک ہے; کیا آپ جانتے ہیں? جی ہاں." in converted


def test_normalize_roman_urdu() -> None:
    """Test Roman Urdu normalization and vowel collapse."""
    raw = "Shukriyaaa! Booohat acha..."
    norm = normalize_roman_urdu(raw, lower=True, remove_punct=True, collapse_vowels=True)
    assert norm == "shukriya bohat acha"


def test_are_roman_urdu_variants() -> None:
    """Test detecting common phonetic/orthographic Roman Urdu variants."""
    assert are_roman_urdu_variants("shukria", "shukriya") is True
    assert are_roman_urdu_variants("shukriya", "shukriyaa") is True
    assert are_roman_urdu_variants("kya", "kia") is True
    assert are_roman_urdu_variants("bohat", "bht") is True
    assert are_roman_urdu_variants("karachi", "lahore") is False
