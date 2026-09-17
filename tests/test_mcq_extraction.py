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


def test_extract_mcq_ambiguous_refusal() -> None:
    """Verify ambiguous, hedged, or multi-choice responses are strictly refused (return None)."""
    # 1. Disjunctions and conjunctions in Urdu
    assert extract_mcq_choice("A یا C") is None
    assert extract_mcq_choice("A اور B") is None
    assert extract_mcq_choice("درست جواب: A یا B") is None
    assert extract_mcq_choice("جواب A یا شاید D ہے") is None

    # 2. Disjunctions and conjunctions in English
    assert extract_mcq_choice("A or C") is None
    assert extract_mcq_choice("Both A and B are correct") is None
    assert extract_mcq_choice("Either C or D") is None
    assert extract_mcq_choice("A / B") is None

    # 3. Hedging between choices
    assert extract_mcq_choice("مجھے لگتا ہے A، لیکن شاید C") is None
    assert extract_mcq_choice("I think maybe A, but perhaps B") is None

    # 4. Multi-choice enumeration without single definitive answer
    assert extract_mcq_choice("A) اسلام آباد B) لاہور") is None
    assert extract_mcq_choice("Option A is interesting, while Option C is also plausible.") is None

    # 5. Multiple conflicting option text mentions
    options = ["اسلام آباد", "لاہور", "کراچی", "پشاور"]
    assert extract_mcq_choice("یہ اسلام آباد بھی ہو سکتا ہے اور لاہور بھی", options=options) is None

    # 6. Disjunctive hedging with explanation
    assert extract_mcq_choice("C یا D کیونکہ دونوں درست معلوم ہوتے ہیں") is None
    assert extract_mcq_choice("میرا حتمی جواب C ہے، اگرچہ A بھی ممکن ہے") is None


def test_extract_mcq_answer_with_explanation() -> None:
    """Verify single unambiguous answer followed by explanation extracts correctly."""
    assert extract_mcq_choice("C کیونکہ سوال میں اسلام آباد کا پوچھا گیا ہے") == "C"
    assert extract_mcq_choice("B. اس کی وجہ یہ ہے کہ زمین سورج کے گرد گھومتی ہے۔") == "B"
