"""Tests for error analysis and failure categorization."""

from urdu_eval.analysis.failures import categorize_failure
from urdu_eval.enums import FailureCategory, Script, TaskType, TranslationDirection
from urdu_eval.models import Sample


def test_categorize_correct() -> None:
    """Test correctly answered samples receive FailureCategory.CORRECT."""
    sample = Sample(id="s1", prompt="سوال", reference="جواب")
    cat = categorize_failure(sample, prediction="جواب", metrics={"exact_match": 1.0})
    assert cat == FailureCategory.CORRECT


def test_categorize_refusal() -> None:
    """Test model refusals in English and Urdu."""
    sample = Sample(id="s2", prompt="سوال", reference="جواب")
    cat_en = categorize_failure(
        sample,
        prediction="I cannot assist with this request.",
        metrics={"exact_match": 0.0, "f1": 0.0},
    )
    assert cat_en == FailureCategory.REFUSAL

    cat_ur = categorize_failure(
        sample,
        prediction="معذرت، میں اس سوال کا جواب دینے سے قاصر ہوں۔",
        metrics={"exact_match": 0.0, "f1": 0.0},
    )
    assert cat_ur == FailureCategory.REFUSAL


def test_categorize_wrong_script() -> None:
    """Test script drift (Latin instead of Urdu, or Urdu instead of Latin)."""
    urdu_sample = Sample(
        id="s3", script=Script.URDU, prompt="پاکستان کا دارالحکومت؟", reference="اسلام آباد"
    )
    cat_latin = categorize_failure(
        urdu_sample,
        prediction="Islamabad",  # English/Latin characters when Urdu script was required
        metrics={"exact_match": 0.0, "f1": 0.0},
    )
    assert cat_latin == FailureCategory.WRONG_SCRIPT

    roman_sample = Sample(
        id="s4", script=Script.ROMAN_URDU, prompt="Darul hukoomat?", reference="Islamabad"
    )
    cat_urdu = categorize_failure(
        roman_sample,
        prediction="اسلام آباد",  # Urdu characters when Roman Urdu script was expected
        metrics={"exact_match": 0.0, "f1": 0.0},
    )
    assert cat_urdu == FailureCategory.WRONG_SCRIPT


def test_categorize_roman_urdu_spelling() -> None:
    """Test recognizing common Roman Urdu orthographic variation."""
    sample = Sample(
        id="s5",
        script=Script.ROMAN_URDU,
        prompt="Kese ho?",
        reference="Shukriya",
    )
    cat = categorize_failure(
        sample,
        prediction="Shukria",
        metrics={"exact_match": 0.0, "f1": 0.0},
    )
    assert cat == FailureCategory.ROMAN_URDU_SPELLING


def test_categorize_task_specific_errors() -> None:
    """Test task-specific errors for reasoning and translation."""
    rs_sample = Sample(id="s6", task=TaskType.REASONING, prompt="5 + 5?", reference="10")
    cat_rs = categorize_failure(rs_sample, prediction="12", metrics={"exact_match": 0.0, "f1": 0.0})
    assert cat_rs == FailureCategory.REASONING_ERROR

    tr_sample = Sample(
        id="s7",
        task=TaskType.TRANSLATION,
        prompt="Translate to English: علم حاصل کرو",
        reference="Seek knowledge",
        direction=TranslationDirection.URDU_TO_ENGLISH,
    )
    cat_tr = categorize_failure(
        tr_sample, prediction="Wrong translation output", metrics={"exact_match": 0.0, "f1": 0.0}
    )
    assert cat_tr == FailureCategory.TRANSLATION_DRIFT
