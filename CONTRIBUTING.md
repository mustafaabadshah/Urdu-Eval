# Contributing to UrduEval

Thank you for your interest in contributing to UrduEval! We welcome contributions from researchers, linguists, software engineers, and community members.

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/urdu-eval/urdu-eval.git
   cd urdu-eval
   ```

2. **Environment & Hatch**:
   UrduEval uses [Hatch](https://hatch.pypa.io/) for environment and dependency management.

   Install Hatch if you do not have it:
   ```bash
   pip install hatch
   ```

3. **Running Quality Checks**:
   ```bash
   # Run tests
   hatch run test

   # Run test coverage
   hatch run cov

   # Run style and lint checks
   hatch run lint

   # Run type checks
   hatch run typecheck

   # Auto-format code
   hatch run fmt
   ```

---

## Contribution Guidelines

### 1. Adding a New Benchmark Adapter
- External benchmarks must have a documented license, citation, and provenance.
- Never copy copyrighted external datasets into the repository; use streaming or loading adapters.
- Provide a small, explicitly labeled 10–20 item development sample for offline integration testing.

### 2. Adding a Model Provider
- Follow the `ModelProvider` protocol defined in `urdu_eval.providers.base`.
- Keep third-party SDK dependencies optional and lazy-loaded so the core package remains lightweight.
- Ensure API keys are read from environment variables and never logged or serialized into result files.

### 3. Adding a Metric
- Implement the `Metric` protocol in `urdu_eval.metrics.base`.
- Register the metric via `@register_metric("metric_name")`.
- Include unit tests verifying deterministic outputs, normalization sensitivity, and edge cases.

---

## Pull Request Process

1. Create a feature branch (`git checkout -b feat/my-feature`).
2. Write tests covering your changes.
3. Ensure `hatch run test`, `hatch run lint`, and `hatch run typecheck` all pass.
4. Submit a descriptive pull request.
