"""Core enumerations for UrduEval."""

from enum import Enum


class Language(str, Enum):
    """Supported language codes."""

    URDU = "ur"
    ENGLISH = "en"
    MIXED = "mixed"


class Script(str, Enum):
    """Supported orthographic scripts."""

    URDU = "urdu"
    ROMAN_URDU = "roman_urdu"
    LATIN = "latin"
    MIXED = "mixed"


class TaskType(str, Enum):
    """Evaluation task types."""

    QA = "qa"
    REASONING = "reasoning"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    INSTRUCTION_FOLLOWING = "instruction_following"
    MMLU = "mmlu"


class TranslationDirection(str, Enum):
    """Translation directions."""

    URDU_TO_ENGLISH = "urdu_to_english"
    ENGLISH_TO_URDU = "english_to_urdu"
    URDU_TO_URDU = "urdu_to_urdu"


class FailureCategory(str, Enum):
    """Failure categorization for error analysis."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    REFUSAL = "refusal"
    HALLUCINATION = "hallucination"
    WRONG_SCRIPT = "wrong_script"
    WRONG_LANGUAGE = "wrong_language"
    TRANSLATION_DRIFT = "translation_drift"
    REASONING_ERROR = "reasoning_error"
    INSTRUCTION_VIOLATION = "instruction_violation"
    ROMAN_URDU_SPELLING = "roman_urdu_spelling"


class NormalizationProfile(str, Enum):
    """Normalization profiles with explicit linguistic guarantees."""

    RAW = "raw"
    CONSERVATIVE = "conservative"
    STANDARD = "standard"
    ROMAN_URDU = "roman_urdu"
