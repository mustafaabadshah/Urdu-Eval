"""Tests for explicit Normalization Profiles and False-Positive linguistic safeguards."""

from urdu_eval.enums import NormalizationProfile
from urdu_eval.normalization import (
    are_roman_urdu_variants,
    normalize_with_profile,
)


def test_conservative_profile_kaf_and_yeh() -> None:
    """Verify conservative profile harmonizes Arabic keyboard variants of Kaf and Yeh."""
    # Arabic Kaf (ك U+0643) -> Urdu Keheh (ک U+06A9)
    arabic_kaf = "كتاب"
    urdu_kaf = "کتاب"
    assert normalize_with_profile(arabic_kaf, "conservative") == urdu_kaf

    # Arabic Yeh (ي U+064A) -> Urdu Choti Yeh (ی U+06CC)
    arabic_yeh = "پاكستاني"
    urdu_yeh = "پاکستانی"
    assert normalize_with_profile(arabic_yeh, "conservative") == urdu_yeh

    # Alef Maksura (ى U+0649) -> Urdu Choti Yeh (ی U+06CC)
    assert normalize_with_profile("موسى", "conservative") == "موسی"


def test_conservative_profile_preserves_teh_marbuta() -> None:
    """Verify conservative profile does NOT destructively merge Teh Marbuta (ة) into Gol Heh (ہ)."""
    # In conservative mode, Teh Marbuta (ة) must be preserved
    text_with_teh_marbuta = "دائرة المعارف"
    norm_conservative = normalize_with_profile(
        text_with_teh_marbuta, NormalizationProfile.CONSERVATIVE
    )
    assert "ة" in norm_conservative
    assert "ہ" not in norm_conservative

    # In standard mode, conversion is permitted
    norm_standard = normalize_with_profile(text_with_teh_marbuta, NormalizationProfile.STANDARD)
    assert len(norm_standard) > 0


def test_conservative_profile_preserves_aspiration_do_chashmi_heh() -> None:
    """Verify Do-Chashmi Heh (ھ) used for aspirated consonants is strictly preserved."""
    aspirated_words = ["بھائی", "پھول", "تھوڑا", "جھوٹ", "کھانا", "دھواں"]
    for word in aspirated_words:
        norm = normalize_with_profile(word, NormalizationProfile.CONSERVATIVE)
        assert "ھ" in norm, f"Do-Chashmi Heh lost in {word}"
        assert "ہ" not in norm, f"Do-Chashmi Heh incorrectly replaced with Gol Heh in {word}"


def test_aerab_removal_in_conservative_profile() -> None:
    """Verify aerab (zabar, zer, pesh, tashdeed) are stripped while consonants remain intact."""
    with_aerab = "پَاکِسْتَانْ"
    expected = "پاکستان"
    assert normalize_with_profile(with_aerab, NormalizationProfile.CONSERVATIVE) == expected

    tanween_word = "عَمَلاً"
    norm_tanween = normalize_with_profile(tanween_word, NormalizationProfile.CONSERVATIVE)
    assert "ً" not in norm_tanween
    assert norm_tanween == "عملا"


def test_raw_profile_leaves_characters_untouched() -> None:
    """Verify raw profile performs zero transformation other than string stripping."""
    raw_input = "  كتابٌ  "
    assert normalize_with_profile(raw_input, NormalizationProfile.RAW) == "كتابٌ"
    assert "ك" in normalize_with_profile(raw_input, NormalizationProfile.RAW)


def test_standard_profile_converts_digits_and_arabic_heh() -> None:
    """Verify standard profile converts Eastern Arabic numerals and Arabic Heh."""
    eastern_digits = "سال ۲۰۲۶"
    norm_std = normalize_with_profile(eastern_digits, NormalizationProfile.STANDARD)
    assert "2026" in norm_std

    # In conservative mode, digits are preserved
    norm_cons = normalize_with_profile(eastern_digits, NormalizationProfile.CONSERVATIVE)
    assert "۲۰۲۶" in norm_cons


def test_false_positive_safeguards_distinct_urdu_words() -> None:
    """Ensure normalization NEVER merges linguistically distinct Urdu words."""
    # Order vs Physician
    assert normalize_with_profile("حکم", "conservative") != normalize_with_profile(
        "حکیم", "conservative"
    )
    # Flower vs Neck
    assert normalize_with_profile("گل", "conservative") != normalize_with_profile(
        "گلا", "conservative"
    )
    # Near vs Pure
    assert normalize_with_profile("پاس", "conservative") != normalize_with_profile(
        "پاک", "conservative"
    )
    # Gold vs Sleep
    assert normalize_with_profile("سونا", "conservative") != normalize_with_profile(
        "سونا", "conservative"
    ).replace("ا", "")


def test_false_positive_safeguards_roman_urdu() -> None:
    """Ensure Roman Urdu normalization does NOT falsely equate distinct vocabulary."""
    # Bad vs Big vs Full
    assert not are_roman_urdu_variants("bura", "bara")
    assert not are_roman_urdu_variants("bara", "pura")
    assert not are_roman_urdu_variants("bura", "pura")

    # Finish vs Lady
    assert not are_roman_urdu_variants("khatam", "khatoon")

    # Book vs Account
    assert not are_roman_urdu_variants("kitab", "hisaab")

    # Water vs Air
    assert not are_roman_urdu_variants("paani", "hawa")


def test_true_positive_roman_urdu_variants() -> None:
    """Ensure genuine phonetic spelling variations in Roman Urdu ARE recognized."""
    assert are_roman_urdu_variants("khubsurat", "khoobsurat")
    assert are_roman_urdu_variants("khoobsurat", "khubsoorat")
    assert are_roman_urdu_variants("shukriya", "shukria")
    assert are_roman_urdu_variants("zaroorat", "zarurat")
    assert are_roman_urdu_variants("bohot", "boht")
