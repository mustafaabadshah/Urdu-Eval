"""UrduEval: Open evaluation infrastructure for Urdu and Roman Urdu AI."""

from urdu_eval.enums import (
    FailureCategory,
    Language,
    Script,
    TaskType,
    TranslationDirection,
)
from urdu_eval.models import (
    BenchmarkMetadata,
    HumanRating,
    JudgeResult,
    LeaderboardEntry,
    MetricResult,
    ModelConfig,
    ModelResponse,
    RunConfig,
    RunMetadata,
    RunResult,
    Sample,
    SampleResult,
)

__version__ = "0.1.2"
__author__ = "UrduEval Contributors"
__license__ = "Apache-2.0"

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "Language",
    "Script",
    "TaskType",
    "TranslationDirection",
    "FailureCategory",
    "Sample",
    "ModelConfig",
    "ModelResponse",
    "BenchmarkMetadata",
    "MetricResult",
    "SampleResult",
    "JudgeResult",
    "RunConfig",
    "RunMetadata",
    "RunResult",
    "LeaderboardEntry",
    "HumanRating",
]
