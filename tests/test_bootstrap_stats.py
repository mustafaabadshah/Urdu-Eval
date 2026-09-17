"""Tests for statistical bootstrap confidence intervals and uncertainty estimation."""

from urdu_eval.enums import CIMethod
from urdu_eval.metrics.stats import (
    bootstrap_confidence_interval,
    compute_metric_ci,
    format_ci_margin,
    format_ci_range,
    wilson_score_interval,
)


def test_bootstrap_empty_and_single() -> None:
    """Verify bootstrap handles empty lists and single items cleanly."""
    assert bootstrap_confidence_interval([]) == (0.0, 0.0)
    assert bootstrap_confidence_interval([0.75]) == (0.75, 0.75)


def test_bootstrap_identical_values() -> None:
    """Verify identical sample values return exact bounds."""
    low, high = bootstrap_confidence_interval([0.8, 0.8, 0.8, 0.8])
    assert low == 0.8
    assert high == 0.8


def test_bootstrap_reproducibility_with_seed() -> None:
    """Verify identical seed yields identical confidence interval."""
    scores = [0.2, 0.5, 0.8, 0.9, 0.4, 0.7, 0.6, 0.85, 0.3, 0.95]
    ci1 = bootstrap_confidence_interval(scores, confidence=0.95, resamples=500, seed=42)
    ci2 = bootstrap_confidence_interval(scores, confidence=0.95, resamples=500, seed=42)
    assert ci1 == ci2
    assert 0.0 <= ci1[0] <= ci1[1] <= 1.0


def test_compute_metric_ci_auto_routing() -> None:
    """Verify compute_metric_ci auto mode routes binary to Wilson and continuous to Bootstrap."""
    binary_scores = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    ci_bin = compute_metric_ci("exact_match", binary_scores, method="auto")
    wilson_expected = wilson_score_interval(6, 8)
    assert abs(ci_bin[0] - wilson_expected[0]) < 1e-4
    assert abs(ci_bin[1] - wilson_expected[1]) < 1e-4

    continuous_scores = [0.45, 0.78, 0.82, 0.91, 0.33, 0.65]
    ci_cont = compute_metric_ci("f1", continuous_scores, method="auto", seed=42)
    boot_expected = bootstrap_confidence_interval(continuous_scores, seed=42)
    assert ci_cont == boot_expected


def test_compute_metric_ci_methods() -> None:
    """Verify explicit methods work as specified."""
    scores = [0.1, 0.4, 0.7, 0.9]
    ci_boot = compute_metric_ci("bleu", scores, method=CIMethod.BOOTSTRAP, seed=42)
    ci_t = compute_metric_ci("bleu", scores, method=CIMethod.STUDENT_T)
    assert 0.0 <= ci_boot[0] <= ci_boot[1] <= 1.0
    assert 0.0 <= ci_t[0] <= ci_t[1] <= 1.0


def test_format_ci_helpers() -> None:
    """Verify string formatting helpers for CI ranges and margins."""
    ci = (0.55, 0.85)
    assert format_ci_range(ci) == "[55.0% - 85.0%]"
    assert format_ci_margin(0.70, ci) == "±15.0%"
