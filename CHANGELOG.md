# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-09-17

### Added
- Explicit Normalization Profiles: `raw`, `conservative` (default), `standard`, and `roman_urdu`.
- Safe conservative normalization: eliminated destructive replacement of Teh Marbuta (`ة` -> `ہ`) and preserved aspiration Do-Chashmi Heh (`ھ`).
- Comprehensive false-positive normalization test suite preventing accidental merging of distinct Urdu vocabulary.
- Statistical 95% Confidence Intervals for all metrics (Wilson score intervals for binomial metrics, Student-t standard error for continuous metrics).
- Transparent side-by-side reporting of unnormalized Raw Exact Match and Normalized Exact Match.
- Task- and metric-aware failure diagnostics replacing universal scalar thresholds.
- UrduMMLU native benchmark adapter supporting MBZUAI UrduMMLU (26,431 questions) across 5 macro-domains (`STEM`, `Humanities`, `Social Sciences`, `Profession`, `Other`).
- `urdu-eval cache stats` and `urdu-eval cache clear` CLI subcommands.
- High-resolution visual assets (`assets/banner.png`, `assets/architecture.png`, `assets/report_preview.png`, and animated `assets/demo.gif`).
- Refined positioning and established official Leaderboard Eligibility Criteria (N >= 500, frozen version, SHA-256 hash, and confidence intervals).

## [0.1.0] - 2026-09-17

### Added
- Core evaluation data models using Pydantic v2.
- Streaming JSONL dataset loader and validation CLI.
- Model provider abstraction protocol with Mock, OpenAI, Anthropic, Ollama, HuggingFace, OpenRouter, and HTTP implementations.
- Comprehensive metric registry: Exact Match, Token F1, SQuAD F1, BLEU, chrF++, ROUGE-1/2/L, and structured LLM-as-Judge with rubric.
- Urdu and Roman Urdu conservative normalization modules.
- Built-in development benchmark suites and custom benchmark loader.
- Evaluation runner with concurrency, disk caching, exponential retry backoff, and checkpoint resume.
- Linguistic error analysis and script drift diagnostics.
- Multi-format reporting: JSON, Markdown, Rich terminal tables, and standalone interactive HTML.
- Multi-run leaderboard aggregator and exporter.
- Unified Typer + Rich CLI (`urdu-eval`).
