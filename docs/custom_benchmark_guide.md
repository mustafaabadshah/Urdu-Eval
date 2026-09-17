# Creating & Evaluating Custom Benchmarks

UrduEval makes evaluating your own custom datasets straightforward. You can create a `.jsonl` file and evaluate any LLM without writing a single line of Python.

---

## 1. Dataset Format (JSON Lines)

Create a file named `my_benchmark.jsonl`. Each line must be a valid JSON object.

### Example Urdu Script Sample
```json
{
  "id": "pk_qa_001",
  "task": "qa",
  "language": "ur",
  "script": "urdu",
  "prompt": "پاکستان کا دارالحکومت کیا ہے؟",
  "reference": "اسلام آباد",
  "metadata": {
    "domain": "geography"
  }
}
```

### Example Roman Urdu Sample
```json
{
  "id": "pk_ru_002",
  "task": "qa",
  "language": "ur",
  "script": "roman_urdu",
  "prompt": "Pakistan ka dar-ul-hukoomat kya hai?",
  "reference": ["Islamabad", "Islam Abad"],
  "metadata": {
    "domain": "geography"
  }
}
```

### Example Translation Sample
```json
{
  "id": "pk_tr_003",
  "task": "translation",
  "language": "ur",
  "script": "urdu",
  "prompt": "Translate to English: علم حاصل کرنا ہر مسلمان پر فرض ہے۔",
  "reference": "Seeking knowledge is an obligation upon every Muslim.",
  "direction": "urdu_to_english"
}
```

---

## 2. Validate Your Dataset

Before running an evaluation, validate your dataset with the CLI:

```bash
urdu-eval validate my_benchmark.jsonl
```

The validator checks:
- UTF-8 encoding integrity
- Valid JSON syntax on every line
- Unique sample IDs
- Presence of required fields (`id`, `prompt`, `reference`)
- Valid task types (`qa`, `reasoning`, `translation`, `summarization`, `mmlu`)
- Script consistency (warns if Roman Urdu contains Arabic script or vice versa)

---

## 3. Run Evaluation

Run your model on your custom dataset:

```bash
# Offline testing with Mock provider
urdu-eval run --provider mock --dataset my_benchmark.jsonl

# Testing OpenAI GPT-4o
urdu-eval run --provider openai --model gpt-4o --dataset my_benchmark.jsonl --metrics exact_match,f1

# Testing Ollama local model
urdu-eval run --provider ollama --model llama3.1:8b --dataset my_benchmark.jsonl
```

---

## 4. Inspect Results & Export HTML Report

```bash
# View sample failures
urdu-eval inspect results/latest/scores.json --failed-only

# Generate interactive HTML report
urdu-eval report results/latest/scores.json --html report.html
```
