# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## API Key & Credential Safety

UrduEval interacts with LLM APIs (OpenAI, Anthropic, OpenRouter, etc.). We follow strict security standards:

1. **No API Key Storage**: API keys are read from environment variables or secure command parameters and are never written to disk, cached files, or serialized evaluation outputs.
2. **No Secret Telemetry**: UrduEval performs zero automatic external telemetry. All evaluation runs, logs, and artifacts are strictly local.
3. **Safe HTTP Client**: Custom endpoints are evaluated with configured timeouts and safe headers.

## Reporting a Vulnerability

If you discover a security vulnerability within UrduEval, please do not file a public issue. Instead, please report it via email to `security@urdueval.org`.

We will acknowledge receipt within 48 hours and provide updates on resolution.
