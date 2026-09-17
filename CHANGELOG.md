# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
