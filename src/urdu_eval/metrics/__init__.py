"""Evaluation metrics for UrduEval."""

from urdu_eval.metrics.base import (
    Metric,
    get_metric,
    list_metrics,
    register_metric,
)
from urdu_eval.metrics.bleu import BLEUMetric
from urdu_eval.metrics.chrf import ChrFMetric
from urdu_eval.metrics.exact_match import ExactMatchMetric
from urdu_eval.metrics.f1 import F1Metric, SquadF1Metric
from urdu_eval.metrics.judge import LLMJudgeMetric, parse_judge_output
from urdu_eval.metrics.mcq import extract_mcq_choice
from urdu_eval.metrics.rouge import ROUGE1Metric, ROUGE2Metric, ROUGELMetric
from urdu_eval.metrics.semantic import SemanticSimilarityMetric
from urdu_eval.metrics.stats import (
    bootstrap_confidence_interval,
    compute_metric_ci,
    continuous_confidence_interval,
    wilson_score_interval,
)

__all__ = [
    "Metric",
    "register_metric",
    "get_metric",
    "list_metrics",
    "ExactMatchMetric",
    "F1Metric",
    "SquadF1Metric",
    "BLEUMetric",
    "ChrFMetric",
    "ROUGE1Metric",
    "ROUGE2Metric",
    "ROUGELMetric",
    "SemanticSimilarityMetric",
    "LLMJudgeMetric",
    "parse_judge_output",
    "extract_mcq_choice",
    "wilson_score_interval",
    "continuous_confidence_interval",
    "bootstrap_confidence_interval",
    "compute_metric_ci",
]
