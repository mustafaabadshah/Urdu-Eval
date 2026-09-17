"""Statistical analysis, confidence interval calculations, and uncertainty estimation."""

from __future__ import annotations

import math


def wilson_score_interval(
    successes: int | float,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate the Wilson score interval for a binomial proportion.

    Preferred over normal approximation for small N or extreme probabilities.
    Returns (lower_bound, upper_bound) clamped to [0.0, 1.0].
    """
    if total <= 0:
        return 0.0, 0.0

    p_hat = successes / total
    # z for 95% = 1.95996 (~1.96)
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
    """Calculate 95% confidence interval for continuous bounded metrics (F1, BLEU, chrF).

    Uses sample standard error (SE) clamped to [0.0, 1.0].
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


def compute_metric_ci(
    metric_name: str,
    scores: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute 95% confidence interval tailored to metric type."""
    if not scores:
        return 0.0, 0.0

    name = metric_name.lower().replace("-", "_")
    is_binary = name in {"exact_match", "accuracy", "script_compliance"}

    if is_binary:
        successes = sum(1 for s in scores if s >= 0.99)
        return wilson_score_interval(successes, len(scores), confidence=confidence)

    return continuous_confidence_interval(scores, confidence=confidence)


def format_ci_range(ci: tuple[float, float]) -> str:
    """Format confidence interval as [lower%, upper%]."""
    return f"[{ci[0]:.1%} - {ci[1]:.1%}]"


def format_ci_margin(mean: float, ci: tuple[float, float]) -> str:
    """Format confidence interval as ± margin percentage."""
    half_width = max(abs(mean - ci[0]), abs(ci[1] - mean))
    return f"±{half_width:.1%}"
