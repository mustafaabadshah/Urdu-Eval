"""Generic HTTP and OpenAI-compatible endpoint provider."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from urdu_eval.models import ModelResponse
from urdu_eval.providers.base import ModelProvider, register_provider


@register_provider("http")
class HTTPProvider(ModelProvider):
    """Generic HTTP endpoint provider supporting OpenAI-compatible or custom REST formats."""

    name = "http"

    def __init__(
        self,
        endpoint: str | None = None,
        model: str = "custom-http-model",
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        headers: dict[str, str] | None = None,
        is_openai_compatible: bool = True,
        **kwargs: Any,
    ) -> None:
        self.endpoint: str = (
            endpoint
            or os.getenv("URDU_EVAL_HTTP_ENDPOINT")
            or "http://localhost:8000/v1/chat/completions"
        )
        self.model = model
        self.api_key = api_key or os.getenv("URDU_EVAL_HTTP_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.headers = headers or {}
        self.is_openai_compatible = is_openai_compatible

        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.headers["Content-Type"] = "application/json"

    @classmethod
    def check_availability(cls) -> tuple[bool, str | None]:
        """HTTP provider is always available as httpx is a core dependency."""
        return True, None

    def _prepare_payload(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1024)

        if self.is_openai_compatible:
            return {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        return {
            "prompt": prompt,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _parse_response(self, data: dict[str, Any], latency_ms: float) -> ModelResponse:
        text = ""
        input_tokens = None
        output_tokens = None
        finish_reason = "stop"

        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                text = choice["message"]["content"] or ""
            elif "text" in choice:
                text = choice["text"] or ""
            finish_reason = choice.get("finish_reason", "stop")

            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
        elif "response" in data:  # e.g., Ollama /api/generate format
            text = data["response"]
            finish_reason = "stop" if data.get("done") else "length"
            input_tokens = data.get("prompt_eval_count")
            output_tokens = data.get("eval_count")
        elif "text" in data:
            text = data["text"]
        elif "output" in data:
            text = str(data["output"])
        else:
            text = str(data)

        return ModelResponse(
            text=text.strip(),
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            metadata={"endpoint": self.endpoint},
        )

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate response via synchronous HTTP POST."""
        payload = self._prepare_payload(prompt, **kwargs)
        start = time.perf_counter()

        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(self.endpoint, json=payload, headers=self.headers)
            resp.raise_for_status()
            latency_ms = (time.perf_counter() - start) * 1000.0
            data = resp.json()

        return self._parse_response(data, latency_ms)

    async def agenerate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate response via asynchronous HTTP POST."""
        payload = self._prepare_payload(prompt, **kwargs)
        start = time.perf_counter()

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(self.endpoint, json=payload, headers=self.headers)
            resp.raise_for_status()
            latency_ms = (time.perf_counter() - start) * 1000.0
            data = resp.json()

        return self._parse_response(data, latency_ms)
