# Benchmarks Registry & Methodology

UrduEval serves as the open evaluation layer for Urdu NLP. This document records the provenance, licensing, citation, task definition, and evaluation methodology for supported benchmarks.

---

## 1. Built-in Development Samples

> [!NOTE]
> The built-in benchmark suites in `urdu_eval/benchmarks/dev_samples/` contain small, carefully curated development samples (10–15 items) designed for development verification, CI testing, and fast sanity checks.
> **They are explicitly NOT intended for scientific benchmark claims.**

### Urdu QA (`urdu-qa`)
- **Task**: Question Answering
- **Language**: Urdu (`ur`)
- **Script**: Urdu script (`urdu`)
- **Metrics**: `exact_match`, `f1`
- **Domain**: Pakistan Geography, History, Culture, and General Knowledge
- **License**: Apache-2.0
- **Version**: 0.1.0

### Urdu Reasoning (`urdu-reasoning`)
- **Task**: Multi-step Logical & Arithmetic Reasoning
- **Language**: Urdu (`ur`)
- **Script**: Urdu script (`urdu`)
- **Metrics**: `exact_match`, `f1`
- **Domain**: Arithmetic word problems, time, geometry, and propositional logic
- **License**: Apache-2.0
- **Version**: 0.1.0

### Urdu Translation (`urdu-translation`)
- **Task**: Machine Translation
- **Languages**: Urdu (`ur`), English (`en`)
- **Directions**: `urdu_to_english`, `english_to_urdu`
- **Metrics**: `bleu`, `chrf++`, `rouge-l`
- **License**: Apache-2.0
- **Version**: 0.1.0

### Urdu Summarization (`urdu-summary`)
- **Task**: Paragraph Summarization
- **Language**: Urdu (`ur`)
- **Script**: Urdu script (`urdu`)
- **Metrics**: `rouge-1`, `rouge-2`, `rouge-l`, `bleu`
- **License**: Apache-2.0
- **Version**: 0.1.0

### Roman Urdu QA (`urdu-roman`)
- **Task**: Transliterated Roman Urdu Question Answering
- **Language**: Urdu (`ur`)
- **Script**: Roman Urdu (`roman_urdu`)
- **Metrics**: `exact_match`, `f1`, `chrf++`
- **License**: Apache-2.0
- **Version**: 0.1.0

### Urdu MMLU Sample (`urdu-mmlu`)
- **Task**: Multiple Choice Domain Knowledge (MMLU)
- **Language**: Urdu (`ur`)
- **Script**: Urdu script (`urdu`)
- **Metrics**: `exact_match`, `accuracy`
- **License**: Apache-2.0
- **Version**: 0.1.0

---

## 2. External Benchmark Adapters

### UrduMMLU (`urdu-mmlu-external`)
- **Full Name**: Massive Multitask Language Understanding in Urdu
- **Source**: HuggingFace Datasets (`UrduMMLU/UrduMMLU`)
- **URL**: [https://huggingface.co/datasets/UrduMMLU/UrduMMLU](https://huggingface.co/datasets/UrduMMLU/UrduMMLU)
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0)
- **Task**: Multiple-choice domain knowledge across 57 academic subjects
- **Evaluation Metric**: Accuracy, Exact Match
- **Adapter**: `urdu_eval.benchmarks.adapters.UrduMMLUAdapter`

### UrduBench Reasoning (`urdu-bench-reasoning`)
- **Full Name**: UrduBench Reasoning Suite
- **Source**: UrduBench Project
- **URL**: [https://github.com/urdu-bench](https://github.com/urdu-bench)
- **License**: Open Access / Academic Research
- **Task**: Complex logical deduction and multi-step reasoning in Urdu
- **Evaluation Metric**: Exact Match, Token F1
- **Adapter**: `urdu_eval.benchmarks.adapters.UrduBenchAdapter`

---

## Evaluation Methodology & Guardrails

1. **Deterministic Default Settings**: Evaluations run with `temperature = 0.0` unless otherwise configured, ensuring reproducible baseline metrics.
2. **No Fabricated Data**: Built-in sample suites are clearly flagged with `is_development_sample = True` and must not be used to substantiate published benchmark claims.
3. **Conservative Normalization**: UrduEval applies character harmonization (Arabic Kaf/Yeh variants to Urdu Unicode) and diacritic stripping without destroying lexical distinctions.
4. **Reproducibility Manifest**: Every evaluation run saves complete system info, model config, prompt hashes, and sample-level outputs in `scores.json` and `samples.jsonl`.
