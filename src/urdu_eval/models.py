"""Core Pydantic models for UrduEval."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from urdu_eval.enums import (
    FailureCategory,
    Language,
    Script,
    TaskType,
    TranslationDirection,
)


class Sample(BaseModel):
    """A single evaluation sample in a dataset."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    task: TaskType = TaskType.QA
    language: Language = Language.URDU
    script: Script = Script.URDU
    prompt: str
    reference: str | list[str]
    options: list[str] | None = None
    direction: TranslationDirection | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Configuration for calling a model."""

    model_config = ConfigDict(extra="allow")

    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    system_prompt: str | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """Structured response from any model provider."""

    model_config = ConfigDict(extra="allow")

    text: str
    model: str
    provider: str
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str = "stop"
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkMetadata(BaseModel):
    """Metadata describing a benchmark suite."""

    id: str
    name: str
    description: str
    languages: list[Language] = Field(default_factory=lambda: [Language.URDU])
    scripts: list[Script] = Field(default_factory=lambda: [Script.URDU])
    tasks: list[TaskType] = Field(default_factory=lambda: [TaskType.QA])
    metrics: list[str] = Field(default_factory=list)
    source: str
    license: str
    version: str
    provenance: str = ""
    is_development_sample: bool = False


class MetricResult(BaseModel):
    """Result of evaluating a single metric."""

    metric_name: str
    score: float
    details: dict[str, Any] = Field(default_factory=dict)


class SampleResult(BaseModel):
    """Result of evaluating a single sample."""

    sample_id: str
    prompt: str
    reference: str | list[str]
    prediction: str
    metrics: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0
    failure_category: FailureCategory | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class JudgeResult(BaseModel):
    """Structured result from an LLM-as-judge evaluation."""

    correctness: int = Field(..., ge=0, le=4, description="Factual and logical correctness (0-4)")
    relevance: int = Field(..., ge=0, le=2, description="Direct relevance to prompt (0-2)")
    language_quality: int = Field(
        ..., ge=0, le=2, description="Fluency and grammatical accuracy (0-2)"
    )
    instruction_following: int = Field(
        ..., ge=0, le=2, description="Compliance with instructions (0-2)"
    )
    total: int = Field(..., ge=0, le=10, description="Sum total out of 10")
    explanation: str = ""
    raw_response: str = ""


class RunConfig(BaseModel):
    """Configuration for an evaluation run."""

    model: ModelConfig
    benchmark_id: str | None = None
    dataset_path: str | None = None
    metrics: list[str] = Field(default_factory=lambda: ["exact_match", "f1"])
    use_cache: bool = True
    workers: int = 1
    max_samples: int | None = None
    output_dir: str = "results"


class RunMetadata(BaseModel):
    """Reproducibility metadata for an evaluation run."""

    model_config = ConfigDict(populate_by_name=True)

    run_id: str
    timestamp: str
    urdu_eval_version: str
    benchmark: BenchmarkMetadata
    dataset_hash: str
    model_settings: ModelConfig = Field(alias="model_config")
    python_version: str
    platform: str


class RunResult(BaseModel):
    """Complete results of an evaluation run."""

    run_id: str
    metadata: RunMetadata
    metrics: dict[str, float]
    samples: list[SampleResult]
    failure_summary: dict[str, int] = Field(default_factory=dict)
    total_samples: int = 0
    failed_samples: int = 0
    mean_latency_ms: float = 0.0


class LeaderboardEntry(BaseModel):
    """An entry in the evaluation leaderboard."""

    model: str
    provider: str
    benchmark_id: str
    benchmark_version: str
    samples: int
    metrics: dict[str, float]
    latency_ms: float
    estimated_cost: float | None = None
    date: str
    run_id: str


class HumanRating(BaseModel):
    """Human evaluation annotation record."""

    sample_id: str
    annotator_id_hash: str
    correctness: float
    fluency: float
    relevance: float
    notes: str = ""
