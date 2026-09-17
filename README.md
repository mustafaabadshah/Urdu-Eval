<div align="center">

<img src="assets/banner.png" alt="UrduEval Banner" width="100%">

<br/>

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.1-blue.svg?logo=pypi&logoColor=white)](https://pypi.org/project/urdu-eval/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-80%20passed-10B981.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen.svg)](tests/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://github.com/python/mypy)

**The open, unified evaluation harness for Urdu and Roman Urdu AI.**

[Quick Start](#-quick-start) • [Live Demo](#-terminal-in-action) • [Architecture](#-architecture) • [Custom Datasets](#-custom-datasets) • [Normalization Profiles](#-linguistic-normalization-profiles) • [UrduMMLU Integration](#-urdummlu-integration--domains) • [Interactive Reports](#-interactive-reports) • [Development Comparison](#-development-run-comparison--leaderboard-protocol)

</div>

---

## 🌟 What is UrduEval?

Urdu is spoken by over **230 million people worldwide**, yet mainstream AI evaluation harnesses treat it as an afterthought. Standard benchmarks fail on Urdu because:
- **Orthographic Inconsistencies**: Variations in Persian/Arabic Kaf (`ک` vs `ك`), Yeh (`ی` vs `ي` vs `ے`), and Heh (`ہ` vs `ھ` vs `ة`).
- **Diacritics & Aerab**: Zabar, Zer, Pesh, Tashdeed are inconsistently present or omitted in digital text.
- **Roman Urdu Orthography**: Millions communicate using Latin script (`"Pakistan aik azeem mulk hai"`), where phonetic spelling varies widely without standard dictionaries (`"khubsurat"` vs `"khoobsurat"` vs `"khobsurat"`).
- **Silent Language Drift**: Models frequently switch to Arabic, Hindi, or English mid-sentence when prompted in Urdu.

Rather than claiming to be a single isolated benchmark, **UrduEval is the open, unified evaluation harness** for Urdu NLP. It bridges established community benchmarks (such as **UrduMMLU**, **UrBLiMP**, and **Urdu Bench**), local LLMs (via Ollama or HuggingFace), cloud APIs (OpenAI, Anthropic, OpenRouter), safe linguistic normalization profiles, statistical confidence intervals, and reproducible diagnostic reports into a single, cohesive CLI and Python library.

---

## 🎬 Terminal in Action

<div align="center">
  <img src="assets/demo.gif" alt="UrduEval CLI Demo" width="90%">
  <p><em>Watch UrduEval evaluate an Ollama model on Urdu QA with live progress, 95% confidence intervals, and error diagnostics.</em></p>
</div>

---

## 🚀 Quick Start

### 1. Installation

Install the lightweight core package with zero heavyweight ML dependencies:

```bash
pip install urdu-eval
```

For your preferred model providers:

```bash
pip install "urdu-eval[ollama]"     # Local Ollama models (Free & Private)
pip install "urdu-eval[openai]"     # OpenAI (GPT-4o, GPT-4o-mini)
pip install "urdu-eval[anthropic]"  # Anthropic (Claude 3.5 Sonnet)
pip install "urdu-eval[hf]"         # Local HuggingFace Transformers
pip install "urdu-eval[all]"        # Install all optional providers
```

Verify your installation:

```bash
urdu-eval --help
```

---

### 2. Inspect Available Benchmarks & Providers

Check built-in benchmarks and active model providers:

```bash
urdu-eval benchmarks
urdu-eval providers
```

| Benchmark ID | Task | Script | Development Samples | Description |
|---|---|---|:---:|---|
| `urdu-qa` | Question Answering | Urdu Script | 15 | Factual QA spanning history, science, geography, and culture |
| `urdu-reasoning` | Multi-step Reasoning | Urdu Script | 12 | Math, syllogisms, and commonsense reasoning in Urdu |
| `urdu-translation` | Bidirectional Translation | Urdu & English | 12 | Urdu-to-English & English-to-Urdu with BLEU and chrF++ |
| `urdu-summary` | Text Summarization | Urdu Script | 10 | News articles and literature summarization |
| `urdu-roman` | Roman Urdu Understanding | Roman Urdu (Latin) | 12 | Conversational Roman Urdu QA and comprehension |
| `urdu-mmlu` | Multi-subject MCQA | Urdu Script | 12 | Curated development sample across humanities and sciences |
| `urdummlu` | Massive Multitask Understanding | Urdu Script | 26,431 | Full MBZUAI UrduMMLU dataset across 5 macro-domains |

> [!NOTE]
> **Methodological Honesty**: Built-in benchmark suites (`urdu-qa`, `urdu-reasoning`, etc.) provide verified **development samples (10–15 items)** designed for smoke testing, developer integration, and continuous integration. They are **not** presented as official academic leaderboards. For formal evaluations, point UrduEval to full community datasets (`urdummlu`) or your own verified `.jsonl` files.

---

### 3. Run Your First Evaluation

#### A. Free Local Models via Ollama (Zero Cost, 100% Private)
Make sure [Ollama](https://ollama.ai) is running locally:

```bash
urdu-eval run --provider ollama --model llama3.1 --benchmark urdu-qa
```

#### B. OpenAI Models
Set your `OPENAI_API_KEY`:

```bash
urdu-eval run --provider openai --model gpt-4o-mini --benchmark urdu-qa
```

#### C. Anthropic Claude
Set your `ANTHROPIC_API_KEY`:

```bash
urdu-eval run --provider anthropic --model claude-3-5-sonnet-20241022 --benchmark urdu-reasoning
```

#### D. OpenRouter (DeepSeek R1, Llama 3.3, Qwen 2.5)
Set your `OPENROUTER_API_KEY`:

```bash
urdu-eval run --provider openrouter --model deepseek/deepseek-r1 --benchmark urdu-translation
```

#### E. Offline Mock Provider (For CI/CD and Testing)
```bash
urdu-eval run --provider mock --benchmark urdu-qa
```

---

## 🏛 UrduMMLU Integration & Domains

UrduEval provides first-class streaming support for the **UrduMMLU** benchmark (MBZUAI, 26,431 verified questions):

```bash
# Run full UrduMMLU dataset (requires 'pip install datasets')
urdu-eval run --benchmark urdummlu --provider ollama --model llama3.1

# Run specific macro-domains
urdu-eval run --benchmark urdummlu-stem --provider openai --model gpt-4o
urdu-eval run --benchmark urdummlu-humanities --provider openai --model gpt-4o
urdu-eval run --benchmark urdummlu-social_sciences --provider openai --model gpt-4o
urdu-eval run --benchmark urdummlu-profession --provider openai --model gpt-4o
```

---

## 🏗 Architecture

UrduEval separates dataset loading, model invocation, linguistic normalization, metric scoring, failure diagnostics, and reporting into clean, decoupled layers:

<div align="center">
  <img src="assets/architecture.png" alt="UrduEval Pipeline Architecture" width="100%">
</div>

---

## 🔤 Linguistic Normalization Profiles

String comparison can artificially depress or inflate LLM scores. UrduEval avoids dangerous global replacements (such as indiscriminately converting Teh Marbuta `ة` $\to$ `ہ`) by providing **explicit, mathematically auditable normalization profiles**:

```bash
# Evaluate with specific normalization profile
urdu-eval run --benchmark urdu-qa --provider ollama --model llama3.1 --normalization conservative
```

| Profile | CLI Flag | Transformations Applied | Best For |
|---|---|---|---|
| **Raw** | `--normalization raw` | Exact string stripping only; no character changes | Strict verbatim benchmarks |
| **Conservative** *(Default)* | `--normalization conservative` | NFC Unicode, Keheh (`ك` $\to$ `ک`), Choti Yeh (`ي` $\to$ `ی`), aerab stripping. **Preserves `ة`, digits, and aspiration `ھ`** | Scientific benchmarks, Academic papers |
| **Standard** | `--normalization standard` | Conservative + Arabic Heh (`ه` $\to$ `ہ`), Eastern Arabic digit conversion (`۰-۹` $\to$ `0-9`), punctuation harmonization | Practical application testing |
| **Roman Urdu** | `--normalization roman_urdu` | Lowercasing, punctuation stripping, vowel elongation collapse (`"bohhht"` $\to$ `"boht"`), phonetic cluster grouping | Roman Urdu chatbots & QA |

### Raw vs. Normalized Metrics Side-by-Side
UrduEval reports unnormalized raw exact match alongside normalized metrics, ensuring complete transparency:

```text
╭────────────────────────────── Evaluation Metrics ──────────────────────────────╮
│ Metric                             Score (Normalized)   95% Confidence Interval│
├────────────────────────────────────────────────────────────────────────────────┤
│ exact_match                                    60.0%         [35.7% - 82.7%]   │
│ raw_exact_match (unnormalized)                 53.3%                       —   │
│ f1                                             78.4%         [58.2% - 91.1%  │
│ chrF++                                         74.2%         [52.8% - 88.0%]   │
╰────────────────────────────────────────────────────────────────────────────────╯
```

---

## 📂 Custom Datasets

Evaluating your own custom Urdu data is a first-class feature in UrduEval. **Zero Python code is required.**

### 1. JSONL Data Format

Prepare a UTF-8 encoded `.jsonl` file:

```json
{"id": "custom-001", "prompt": "علامہ اقبال کا تعلق کس شہر سے تھا؟", "reference": "سیالکوٹ", "task": "qa", "script": "urdu"}
{"id": "custom-002", "prompt": "Pakistan ka qaumi khel konsa hai?", "reference": "Hockey", "task": "qa", "script": "roman_urdu"}
{"id": "custom-003", "prompt": "درج ذیل جملے کا انگریزی میں ترجمہ کریں: محنت میں عظمت ہے۔", "reference": "There is dignity in hard work.", "task": "translation", "script": "urdu"}
```

### 2. Validate Dataset Before Running

```bash
urdu-eval validate my_dataset.jsonl
```

### 3. Run Evaluation on Your Dataset

```bash
urdu-eval run \
  --provider ollama \
  --model llama3.1 \
  --dataset my_dataset.jsonl \
  --metrics exact_match,f1,chrf \
  --workers 4
```

---

## 📊 Comprehensive Metrics & 95% Confidence Intervals

Every metric reported by UrduEval includes **95% Confidence Intervals** (Wilson score intervals for binomial metrics and sample standard error intervals for continuous metrics), making statistical uncertainty explicit:

| Metric | CLI Flag | Best For | Description |
|---|---|---|---|
| **Exact Match** | `exact_match` | QA, MCQA | Normalized string equality check with Wilson 95% CI |
| **Token F1** | `f1` | QA, Extraction | Harmonic mean of token precision and recall with Urdu punctuation tokenization |
| **chrF / chrF++** | `chrf` | Translation, Generation | Character n-gram F-score with word 2-grams (recommended for morphologically rich languages like Urdu) |
| **BLEU-4** | `bleu` | Translation | Standard 1-to-4 n-gram precision with brevity penalty |
| **ROUGE-L** | `rouge-l` | Summarization | Longest Common Subsequence (LCS) overlap score |
| **LLM-as-a-Judge** | `judge` | Open-Ended, Reasoning | Structured rubric scoring (0.0 to 1.0) with multi-attribute criteria and JSON verification |

---

## 🔍 Task-Specific Failure Diagnostics

Rather than arbitrary universal thresholds, UrduEval uses task-aware diagnostic classification:

```text
                              Sample Evaluation Outcome
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
         Pass Threshold                                  Fail Threshold
                 │                                               │
             [Correct]                    ┌──────────────────────┴──────────────────────┐
                                          ▼                                             ▼
                                    Model Refusal                               Linguistic Slip
                                  ("I cannot...", etc.)                       (Script / Drift)
                                          │                                             │
                                      [Refusal]                                  [Wrong Script]
                                                                                        │
                                                          ┌─────────────────────────────┴─────────────────────────────┐
                                                          ▼                                                           ▼
                                                  Translation Drift                                           Reasoning Error
                                            (Target language mismatch)                                     (Math / logic step error)
```

---

## 📑 Interactive Reports

Generate a self-contained, interactive HTML report with search filters, KPI cards, and sample-level inspection:

<div align="center">
  <img src="assets/report_preview.png" alt="UrduEval Interactive HTML Report Preview" width="100%">
</div>

```bash
# Generate report for a run
urdu-eval report results/run_20260917_urdu_qa/scores.json --html results/report.html
```

---

## 📊 Development Run Comparison & Leaderboard Protocol

### Sample Verification Results (Development Suite)
The table below illustrates sample verification results on the built-in development suite ($N=15$). Notice how the **95% Confidence Intervals** clearly reveal sample size uncertainty:

| Model | Provider | Benchmark | Samples | EXACT_MATCH (95% CI) | Token F1 | Mean Latency |
|---|---|---|:---:|:---:|:---:|:---:|
| `gpt-4o` | openai | urdu-qa (v0.1.0) | 15 | **80.0%** `[54.8% - 93.0%]` | 89.2% | 320 ms |
| `claude-3-5-sonnet` | anthropic | urdu-qa (v0.1.0) | 15 | **73.3%** `[48.1% - 89.1%]` | 86.1% | 410 ms |
| `llama3.1:8b` | ollama | urdu-qa (v0.1.0) | 15 | **60.0%** `[35.7% - 82.7%]` | 78.4% | 142 ms |

> [!IMPORTANT]
> **Official Leaderboard Eligibility Criteria**:
> In UrduEval, a benchmark evaluation is only certified for public leaderboard ranking if it satisfies:
> 1. **Sample Size**: $N \ge 500$ verified test items.
> 2. **Dataset Version**: Frozen version with recorded SHA-256 cryptographic provenance hash.
> 3. **Fixed Hyperparameters**: Deterministic decoding (`temperature=0.0`, fixed seed where supported).
> 4. **Statistical Rigor**: 95% Confidence Intervals reported for all primary metrics.
> 5. **Diagnostic Transparency**: Full error taxonomy distribution published alongside scalar scores.

---

## ⚡ Cache Management

UrduEval features a persistent SQLite cache to prevent redundant API invocations and cost:

```bash
# View cache statistics and database size
urdu-eval cache stats

# Clear response cache
urdu-eval cache clear
```

---

## 🐍 Python Library Usage

UrduEval can be imported directly into Python scripts:

```python
from urdu_eval.benchmarks import get_benchmark
from urdu_eval.models import ModelConfig, RunConfig
from urdu_eval.runner import EvaluationRunner

# 1. Configure model and run settings
model_cfg = ModelConfig(provider="ollama", model="llama3.1", temperature=0.0)
run_cfg = RunConfig(
    model=model_cfg,
    benchmark_id="urdu-qa",
    normalization_profile="conservative",
    metrics=["exact_match", "f1", "chrf"],
    workers=4,
)

# 2. Load benchmark & execute
benchmark = get_benchmark("urdu-qa")
runner = EvaluationRunner(config=run_cfg, benchmark=benchmark)
result = runner.run()

# 3. Access structured results and confidence intervals
print(f"Total Samples: {result.total_samples}")
print(f"Exact Match:   {result.metrics['exact_match']:.1%}")
if "exact_match" in result.confidence_intervals:
    low, high = result.confidence_intervals["exact_match"]
    print(f"95% CI:        [{low:.1%} - {high:.1%}]")
```

---

## 🛠 Command Reference

| Command | Purpose |
|---|---|
| `urdu-eval run` | Execute benchmark evaluation against a target model |
| `urdu-eval validate <file.jsonl>` | Validate custom dataset schema, encoding, and script consistency |
| `urdu-eval benchmarks` | List all registered built-in and external benchmarks |
| `urdu-eval providers` | Check availability and prerequisites for model providers |
| `urdu-eval metrics` | List available metrics and supported parameters |
| `urdu-eval check` | Test model connectivity and verify API authentication |
| `urdu-eval compare <run_a> <run_b>` | Generate side-by-side metric diff between two evaluation runs |
| `urdu-eval inspect <scores.json>` | Inspect individual sample prompts, answers, and error categories |
| `urdu-eval report <scores.json>` | Generate standalone interactive HTML or Markdown reports |
| `urdu-eval leaderboard` | Aggregate runs across models into a sorted comparative leaderboard |
| `urdu-eval cache stats` | View cache hit counts, entries, and database disk usage |
| `urdu-eval cache clear` | Clear SQLite response cache |
| `urdu-eval clean` | Remove past test run results and reset local leaderboard |
| `urdu-eval experiment <config.yaml>` | Run automated multi-model multi-benchmark experiment pipeline |

---

## 📜 Citation

If you use UrduEval in your academic work, research, or product development, please cite:

```bibtex
@software{urdu_eval2026,
  author = {UrduEval Contributors},
  title = {UrduEval: Open Evaluation Layer for Urdu and Roman Urdu AI},
  year = {2026},
  url = {https://github.com/mustafaabadshah/Urdu-Eval}
}
```

---

## 📄 License

UrduEval is distributed under the open-source **[Apache License 2.0](LICENSE)**.
