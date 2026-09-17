"""Structured LLM-as-Judge evaluation metric with Pydantic rubric validation."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from urdu_eval.metrics.base import Metric, register_metric
from urdu_eval.models import JudgeResult, MetricResult
from urdu_eval.providers.base import ModelProvider

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for Urdu and Roman Urdu NLP.
Evaluate the model response against the reference and prompt.

Prompt:
{prompt}

Reference:
{reference}

Model Response:
{prediction}

Evaluate strictly according to this rubric:
1. correctness (0-4): Factual and logical accuracy (0 = completely wrong, 4 = perfectly accurate).
2. relevance (0-2): Direct relevance to the question (0 = irrelevant, 2 = directly relevant).
3. language_quality (0-2): Fluency, grammar, and script naturalness (0 = broken/unintelligible, 2 = natural/grammatical).
4. instruction_following (0-2): Compliance with constraints/format (0 = ignored instructions, 2 = fully followed).
5. total (0-10): Sum of the four scores above.
6. explanation: A concise 1-2 sentence justification.

Return ONLY a JSON object with this exact structure:
{{
  "correctness": 4,
  "relevance": 2,
  "language_quality": 2,
  "instruction_following": 2,
  "total": 10,
  "explanation": "..."
}}
"""

JSON_BLOCK_REGEX = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Robustly extract a JSON object from text or markdown codeblocks."""
    text = raw_text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Try regex extracting markdown code block
    match = JSON_BLOCK_REGEX.search(text)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Try finding outermost { and }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Regex fallback extraction of numbers
    fallback: dict[str, Any] = {}
    for key in ["correctness", "relevance", "language_quality", "instruction_following", "total"]:
        num_match = re.search(rf'"{key}"\s*:\s*(\d+)', text, re.IGNORECASE)
        if num_match:
            fallback[key] = int(num_match.group(1))

    if "correctness" in fallback:
        return fallback

    raise ValueError(f"Unable to parse structured JSON from judge output: {raw_text[:200]}")


def parse_judge_output(raw_text: str) -> JudgeResult:
    """Parse and validate LLM judge output using JudgeResult schema."""
    try:
        data = extract_json_payload(raw_text)
    except Exception:
        # Graceful fallback on total parse failure
        return JudgeResult(
            correctness=0,
            relevance=0,
            language_quality=0,
            instruction_following=0,
            total=0,
            explanation="Failed to parse structured JSON from judge output.",
            raw_response=raw_text,
        )

    # Clamp scores to rubric boundaries
    c = max(0, min(4, int(data.get("correctness", 0))))
    r = max(0, min(2, int(data.get("relevance", 0))))
    lq = max(0, min(2, int(data.get("language_quality", 0))))
    inf = max(0, min(2, int(data.get("instruction_following", 0))))
    total = max(0, min(10, c + r + lq + inf))
    explanation = str(data.get("explanation", ""))

    try:
        return JudgeResult(
            correctness=c,
            relevance=r,
            language_quality=lq,
            instruction_following=inf,
            total=total,
            explanation=explanation,
            raw_response=raw_text,
        )
    except ValidationError:
        return JudgeResult(
            correctness=0,
            relevance=0,
            language_quality=0,
            instruction_following=0,
            total=0,
            explanation="Validation error during judge output parsing.",
            raw_response=raw_text,
        )


@register_metric("llm_judge")
@register_metric("judge")
class LLMJudgeMetric(Metric):
    """LLM-as-judge metric using structured rubric evaluation."""

    name = "llm_judge"

    def __init__(
        self,
        judge_provider: ModelProvider | None = None,
        prompt_version: str = "v1",
    ) -> None:
        self.judge_provider = judge_provider
        self.prompt_version = prompt_version

    def build_judge_prompt(self, prompt: str, reference: str | list[str], prediction: str) -> str:
        ref_text = reference if isinstance(reference, str) else " | ".join(reference)
        return JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt,
            reference=ref_text,
            prediction=prediction,
        )

    def evaluate_with_judge(
        self, prompt: str, reference: str | list[str], prediction: str
    ) -> JudgeResult:
        if self.judge_provider is None:
            raise ValueError("Judge provider not configured for LLMJudgeMetric.")

        judge_prompt = self.build_judge_prompt(prompt, reference, prediction)
        resp = self.judge_provider.generate(judge_prompt, temperature=0.0)
        return parse_judge_output(resp.text)

    def compute(self, prediction: str, reference: str | list[str], **kwargs: Any) -> float:
        """Compute normalized judge score (0.0 to 1.0)."""
        prompt = kwargs.get("prompt", "")
        if self.judge_provider is None:
            # Fallback for offline calculation or pre-supplied judge output
            raw_output = kwargs.get("judge_raw_output")
            if raw_output:
                res = parse_judge_output(raw_output)
                return res.total / 10.0
            return 0.0

        res = self.evaluate_with_judge(prompt, reference, prediction)
        return res.total / 10.0

    def compute_details(
        self, prediction: str, reference: str | list[str], **kwargs: Any
    ) -> MetricResult:
        prompt = kwargs.get("prompt", "")
        raw_output = kwargs.get("judge_raw_output")

        if self.judge_provider is not None:
            res = self.evaluate_with_judge(prompt, reference, prediction)
        elif raw_output:
            res = parse_judge_output(raw_output)
        else:
            res = JudgeResult(
                correctness=0,
                relevance=0,
                language_quality=0,
                instruction_following=0,
                total=0,
                explanation="No judge provider or judge raw output provided.",
            )

        return MetricResult(
            metric_name=self.name,
            score=res.total / 10.0,
            details={
                "correctness": res.correctness,
                "relevance": res.relevance,
                "language_quality": res.language_quality,
                "instruction_following": res.instruction_following,
                "total_10": res.total,
                "explanation": res.explanation,
                "prompt_version": self.prompt_version,
            },
        )
