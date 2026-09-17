"""Tests for deterministic MCQ answer extraction protocol."""

from urdu_eval.metrics.mcq import extract_mcq_choice


def test_extract_mcq_bare_letters() -> None:
    """Verify single bare letters are extracted properly."""
    assert extract_mcq_choice("A") == "A"
    assert extract_mcq_choice("b") == "B"
    assert extract_mcq_choice("C.") == "C"
    assert extract_mcq_choice("(D)") == "D"
    assert extract_mcq_choice("[A]") == "A"


def test_extract_mcq_urdu_prefixes() -> None:
    """Verify natural Urdu prefixes and conversational responses."""
    assert extract_mcq_choice("جواب C ہے") == "C"
    assert extract_mcq_choice("درست جواب: A") == "A"
    assert extract_mcq_choice("صحیح ترین آپشن: D") == "D"
    assert extract_mcq_choice("میرے خیال میں درست جواب C ہے کیونکہ یہ صحیح ہے۔") == "C"
    assert extract_mcq_choice("درست انتخاب: (B)") == "B"
    assert extract_mcq_choice("A صحیح ہے") == "A"
    assert extract_mcq_choice("آپشن B درست ہوگا") == "B"


def test_extract_mcq_english_prefixes() -> None:
    """Verify English answer prefix patterns."""
    assert extract_mcq_choice("The correct option is: B") == "B"
    assert extract_mcq_choice("Answer: C") == "C"
    assert extract_mcq_choice("Choice: (D)") == "D"


def test_extract_mcq_option_text_fallback() -> None:
    """Verify option text fallback when choice letter is omitted."""
    options = ["اسلام آباد", "لاہور", "کراچی", "پشاور"]
    pred = "پاکستان کا دارالحکومت اسلام آباد ہے"
    assert extract_mcq_choice(pred, options=options) == "A"

    pred2 = "صحیح شہر لاہور ہے"
    assert extract_mcq_choice(pred2, options=options) == "B"


def test_extract_mcq_unextractable() -> None:
    """Verify empty or non-matching text returns None."""
    assert extract_mcq_choice("") is None
    assert extract_mcq_choice("مجھے معلوم نہیں ہے") is None
