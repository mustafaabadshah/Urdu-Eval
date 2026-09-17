"""Statistical analysis, confidence interval calculations, and uncertainty estimation."""

from __future__ import annotations

import math
import random

from urdu_eval.enums import CIMethod


def wilson_score_interval(
    successes: int | float,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate the Wilson score interval for a binomial proportion.

    Recommended for binomial classification metrics (Exact Match, Accuracy).
    Returns (lower_bound, upper_bound) clamped to [0.0, 1.0].
    """
    if total <= 0:
        return 0.0, 0.0

    p_hat = successes / total
    z = (
        1.95996
        if abs(confidence - 0.95) < 0.01
        else 2.576
        if abs(confidence - 0.99) < 0.01
        else 1.645
    )

    denominator = 1.0 + (z**2 / total)
    centre = (p_hat + (z**2 / (2 * total))) / denominator
    spread = (z / denominator) * math.sqrt(
        (p_hat * (1.0 - p_hat) / total) + (z**2 / (4 * total**2))
    )

    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return lower, upper


def continuous_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate Student-t standard error confidence interval for continuous bounded metrics.

    Parametric approximation; assumes asymptotic normality.
    """
    n = len(scores)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        val = max(0.0, min(1.0, scores[0]))
        return val, val

    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
    std_err = math.sqrt(variance / n)

    z = 1.95996 if abs(confidence - 0.95) < 0.01 else 2.576
    margin = z * std_err

    lower = max(0.0, mean - margin)
    upper = min(1.0, mean + margin)
    return lower, upper


def bootstrap_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Non-parametric percentile bootstrap confidence interval for continuous/bounded scores.

    Resamples with replacement from the observed sample scores to construct
    an empirical sampling distribution without assuming normality.
    Deterministically seeded for reproducible scientific reporting.
    """
    n = len(scores)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        val = max(0.0, min(1.0, scores[0]))
        return val, val

    # If all values are identical, return exact bounds
    if all(s == scores[0] for s in scores):
        val = max(0.0, min(1.0, scores[0]))
        return val, val

    rng = random.Random(seed)
    bootstrap_means: list[float] = []

    for _ in range(resamples):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()

    alpha = (1.0 - confidence) / 2.0
    lower_idx = int(alpha * resamples)
    upper_idx = min(resamples - 1, int((1.0 - alpha) * resamples) - 1)

    lower = max(0.0, min(1.0, bootstrap_means[lower_idx]))
    upper = max(0.0, min(1.0, bootstrap_means[upper_idx]))
    return lower, upper


def compute_metric_ci(
    metric_name: str,
    scores: list[float],
    confidence: float = 0.95,
    method: str | CIMethod = "auto",
    resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute confidence interval tailored to metric type and requested method.

    Methods:
    - 'auto' (default): Wilson score for binomial metrics (Exact Match, Accuracy);
      non-parametric percentile bootstrap for continuous metrics (F1, BLEU, chrF++, ROUGE).
    - 'bootstrap': Bootstrap resampling for all metrics.
    - 'wilson': Wilson score for binary metrics; fallback to bootstrap for continuous.
    - 't': Parametric Student-t SE approximation.
    """
    if not scores:
        return 0.0, 0.0

    m_method = str(method.value if isinstance(method, CIMethod) else method).lower()
    name = metric_name.lower().replace("-", "_")
    is_binary = name in {"exact_match", "accuracy", "script_compliance"}

    if m_method == "wilson":
        if is_binary:
            successes = sum(1 for s in scores if s >= 0.99)
            return wilson_score_interval(successes, len(scores), confidence=confidence)
        return bootstrap_confidence_interval(
            scores, confidence=confidence, resamples=resamples, seed=seed
        )

    if m_method == "t":
        if is_binary:
            successes = sum(1 for s in scores if s >= 0.99)
            return wilson_score_interval(successes, len(scores), confidence=confidence)
        return continuous_confidence_interval(scores, confidence=confidence)

    if m_method == "bootstrap":
        return bootstrap_confidence_interval(
            scores, confidence=confidence, resamples=resamples, seed=seed
        )

    # Default 'auto': Wilson for binary, bootstrap for continuous
    if is_binary:
        successes = sum(1 for s in scores if s >= 0.99)
        return wilson_score_interval(successes, len(scores), confidence=confidence)

    return bootstrap_confidence_interval(
        scores, confidence=confidence, resamples=resamples, seed=seed
    )


def format_ci_range(ci: tuple[float, float]) -> str:
    """Format confidence interval as [lower%, upper%]."""
    return f"[{ci[0]:.1%} - {ci[1]:.1%}]"


def format_ci_margin(mean: float, ci: tuple[float, float]) -> str:
    """Format confidence interval as ± margin percentage."""
    half_width = max(abs(mean - ci[0]), abs(ci[1] - mean))
    return f"±{half_width:.1%}"
