# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-17

### Added
- **Statistical Bootstrap Confidence Intervals**: Non-parametric percentile bootstrap confidence intervals (1,000 resamples, deterministically seeded) for all continuous and bounded metrics (`f1`, `bleu`, `chrf++`, `rouge-l`, `judge`), complementing Wilson score intervals for binomial metrics. Configurable via `--ci-method auto|bootstrap|wilson|t`.
- **Reproducibility Audit Engine (`urdu-eval reproduce <manifest>`)**: Complete CLI verification checking dataset cryptographic hash, benchmark version, prompt protocol, model hyperparameters, normalization profile, and UrduEval version compatibility.
- **Benchmark Integrity Verification (`urdu-eval benchmark verify <id>`)**: Automated dataset audit checking sample count, schema validity, field completeness, duplicate prompts, and cryptographic SHA-256 provenance.
- **UrBLiMP Linguistic Minimal Pairs Adapter**: Native adapter for the 5,696-pair UrBLiMP benchmark across 10 morphosyntactic phenomena (subject-verb agreement, case marking, word order, converb agreement, anaphora binding, etc., with 96.1% human agreement) plus bundled offline development samples.
- **UrduMMLU Complete 5-Domain Exposure**: Fully registered and exposed `urdummlu-other` alongside `stem`, `humanities`, `social_sciences`, and `profession`.
- **Deterministic MCQ Extraction Protocol**: Robust choice-letter extraction (`A`, `B`, `C`, `D`) handling common conversational and natural Urdu model outputs (e.g. `جواب C ہے`, `درست جواب: (A)`, `صحیح آپشن: B`).
- **Canonical Request Caching**: SHA-256 caching encompassing all generation determinants (`temperature`, `top_p`, `max_tokens`, `seed`, `system_prompt`, `benchmark_id`, `prompt_template_version`) to eliminate silent cache contamination.
- **Roman Urdu Strict vs. Phonetic Profiles**: Formal separation between `roman_urdu_strict` (zero vowel alteration) and `roman_urdu_phonetic` (safe 3+ vowel elongation collapse and diagnostic cluster matching).
- **Attribution & Scholarly Metadata**: Formally attributed Syed Mustafa Badshah as primary author in `CITATION.cff`, `pyproject.toml`, and README with academic speaker demographics citation.

## [0.1.2] - 2026-09-17

### Fixed
- Fixed Urdu text shaping and font rendering in documentation visual assets (`assets/banner.png` and `assets/report_preview.png`). Every Urdu word now connects properly using contextual OpenType cursive letterforms (initial, medial, final, and isolated glyphs) via `arabic-reshaper` and `python-bidi`.
- Eliminated bidirectional parenthesis mirroring anomalies in report preview cards by introducing clean pipe (`|`) field separators and dynamic font boundary metrics.
- Re-spaced bottom KPI cards on `assets/banner.png` to prevent text truncation or overlap across columns.

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
