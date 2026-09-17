# UrduEval

**Measure how well AI understands Urdu.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type_checked-mypy-blue.svg)](https://github.com/python/mypy)

An open-source evaluation toolkit for testing LLMs on Urdu script, Roman Urdu, translation, reasoning, summarization, instruction following, and real-world Pakistani language use.

---

## What is UrduEval?

UrduEval is **not** merely a single benchmark or a collection of test prompts. UrduEval is the **open evaluation layer** for Urdu NLP.

```text
       ┌──────────────┐
       │ Target Model │ (API / Ollama / HF / HTTP)
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │   UrduEval   │ (Harness & Concurrency)
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │  Benchmarks  │ (Built-in / UrduMMLU / Custom JSONL)
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │Normalization │ (Conservative Urdu & Roman Urdu)
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │   Metrics    │ (Exact Match, F1, BLEU, chrF++, Judge)
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │Error Analysis│ (Script drift, Factual, Refusal)
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │Report/Leader │ (JSON, Terminal, Standalone HTML)
       └──────────────┘
```

Researchers, developers, and teams can:
1. Run curated benchmark suites or established external datasets.
2. Evaluate custom datasets with zero Python code via streaming JSONL.
3. Test remote API models (OpenAI, Anthropic, OpenRouter, Custom HTTP) and local models (Ollama, HuggingFace).
4. Accurately measure both **Urdu script** (`پاکستان ایک خوبصورت ملک ہے`) and **Roman Urdu** (`Pakistan aik khoobsurat mulk hai`).
5. Inspect individual sample failures, language drift, and script consistency.
6. Generate standalone, publication-ready HTML reports and reproducible JSON results.

---

## Quick Start

### 1. Installation

```bash
pip install urdu-eval
```

For specific model providers:
```bash
pip install "urdu-eval[openai]"     # OpenAI models
pip install "urdu-eval[anthropic]"  # Anthropic Claude models
pip install "urdu-eval[hf]"         # HuggingFace transformers
pip install "urdu-eval[ollama]"     # Ollama local models
pip install "urdu-eval[all]"        # All providers
```

### 2. Check Available Benchmarks

```bash
urdu-eval benchmarks
```

### 3. Run an Evaluation

```bash
# Using Mock provider (no API key needed)
urdu-eval run --provider mock --benchmark urdu-qa

# Using OpenAI
urdu-eval run --provider openai --model gpt-4o --benchmark urdu-qa

# Using Ollama
urdu-eval run --provider ollama --model llama3.1:8b --benchmark urdu-reasoning
```

### 4. Evaluate Your Own Dataset

Validate dataset format:
```bash
urdu-eval validate my_dataset.jsonl
```

Run evaluation:
```bash
urdu-eval run --model gpt-4o --dataset my_dataset.jsonl --metrics exact_match,f1
```

### 5. Generate Standalone HTML Reports

```bash
urdu-eval report results/run_xxx/scores.json --html report.html
```

---

## Supported Language Forms

UrduEval explicitly distinguishes and reports:
- **Urdu Script**: Arabic/Perso-Arabic script (Nastaliq & Naskh conventions).
- **Roman Urdu**: Latin-script transliterations with phonetic and orthographic variations.
- **Code-Mixing**: Mixed Urdu-English conversational phrasing.
- **Translation Directions**: `urdu_to_english`, `english_to_urdu`, and monolingual `urdu_to_urdu`.

---

## Architecture & Design Principles

- **No Fabricated Benchmarks**: Built-in suites provide clearly labeled development samples (10–20 items) for workflow testing, while production evaluations connect to established public datasets or custom user benchmarks.
- **Lazy Imports**: The core package remains lightweight and does not enforce PyTorch or heavyweight dependencies.
- **Reproducibility**: Every evaluation records run ID, dataset hash, model configuration, prompt templates, and system versions.
- **Resumable & Cached**: Network errors or interruptions resume seamlessly from disk checkpoints without duplicate API costs.

---

## Citation

If you use UrduEval in your academic research, please cite:

```bibtex
@software{urdu_eval2026,
  author = {UrduEval Contributors},
  title = {UrduEval: Open Evaluation Infrastructure for Urdu and Roman Urdu AI},
  year = {2026},
  url = {https://github.com/urdu-eval/urdu-eval}
}
```

---

## License

UrduEval is open-source software licensed under the [Apache License 2.0](LICENSE).
