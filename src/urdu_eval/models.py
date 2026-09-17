"""Core Pydantic models for UrduEval."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from urdu_eval.enums import (
    CIMethod,
    ContaminationStatus,
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
    seed: int | None = 42
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
    dataset_sha256: str = ""
    citation: str = ""
    is_development_sample: bool = False
    dataset_scope: str = "official"  # "official" or "development"
    official_benchmark_size: int | None = None


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


class PromptProtocol(BaseModel):
    """Prompt construction protocol for reproducible evaluation across runs."""

    model_config = ConfigDict(extra="allow")

    language: str = "urdu"
    template_version: str = "1.0"
    few_shot: int = 0
    system_prompt: str | None = None
    answer_format: str = "choice_letter"
    answer_extraction_version: str = "mcq-v1"


class ContaminationInfo(BaseModel):
    """Benchmark training-data contamination tracking."""

    status: ContaminationStatus = ContaminationStatus.UNKNOWN
    method: str | None = None
    details: str = ""


class CIConfig(BaseModel):
    """Statistical confidence interval configuration."""

    method: CIMethod = CIMethod.AUTO
    confidence: float = 0.95
    resamples: int = 1000
    seed: int = 42


class RunConfig(BaseModel):
    """Configuration for an evaluation run."""

    model: ModelConfig
    benchmark_id: str | None = None
    dataset_path: str | None = None
    metrics: list[str] = Field(default_factory=lambda: ["exact_match", "f1"])
    normalization_profile: str = "conservative"
    use_cache: bool = True
    workers: int = 1
    max_samples: int | None = None
    output_dir: str = "results"
    ci_config: CIConfig = Field(default_factory=CIConfig)
    prompt_protocol: PromptProtocol = Field(default_factory=PromptProtocol)
    contamination: ContaminationInfo = Field(default_factory=ContaminationInfo)


class RunMetadata(BaseModel):
    """Reproducibility metadata for an evaluation run."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    run_id: str
    timestamp: str
    urdu_eval_version: str
    benchmark: BenchmarkMetadata

    # Explicit Benchmark & Dataset Provenance
    benchmark_id: str = ""
    benchmark_version: str = "1.0.0"
    dataset_hash: str = ""
    dataset_sha256: str = ""
    dataset_source: str = ""
    dataset_provenance: str = ""
    dataset_size: int = 0
    dataset_scope: str = "official"  # "official" or "development"
    is_official_evaluation: bool = True
    official_benchmark_size: int | None = None

    # Model & Execution Hyperparameters
    model_settings: ModelConfig = Field(alias="model_config")
    model: str = ""
    provider: str = ""
    model_revision: str = "default"
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None

    # Prompt Protocol
    prompt_protocol: PromptProtocol = Field(default_factory=PromptProtocol)
    prompt_template_version: str = "1.0"
    prompt_language: str = "urdu"
    shot_count: int = 0
    answer_extraction_version: str = "mcq-v1"

    # Normalization & Metrics
    normalization_profile: str = "conservative"
    normalization_version: str = "v1.0"
    metrics: list[str] = Field(default_factory=list)

    # Statistical Configuration
    ci_config: CIConfig = Field(default_factory=CIConfig)
    ci_method: str = "auto"
    ci_resamples: int = 1000
    ci_seed: int = 42

    # Contamination & Environment
    contamination: ContaminationInfo = Field(default_factory=ContaminationInfo)
    contamination_status: str = "unknown"
    python_version: str = ""
    platform: str = ""


class RunResult(BaseModel):
    """Complete results of an evaluation run."""

    run_id: str
    metadata: RunMetadata
    metrics: dict[str, float]
    raw_metrics: dict[str, float] = Field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = Field(default_factory=dict)
    ci_config: CIConfig = Field(default_factory=CIConfig)
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
    raw_metrics: dict[str, float] = Field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = Field(default_factory=dict)
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
