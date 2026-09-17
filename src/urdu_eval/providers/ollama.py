"""Ollama local model provider using native REST API."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from urdu_eval.models import ModelResponse
from urdu_eval.providers.base import ModelProvider, register_provider


@register_provider("ollama")
class OllamaProvider(ModelProvider):
    """Local Ollama provider (Llama-3.1, Qwen, Mistral, etc.)."""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.1",
        host: str | None = None,
        timeout_seconds: float = 120.0,
        **kwargs: Any,
    ) -> None:
        self.model = model
        raw_host: str = host or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        self.host: str = raw_host.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @classmethod
    def check_availability(cls) -> tuple[bool, str | None]:
        host: str = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(f"{host}/api/tags")
                if r.status_code == 200:
                    return True, None
                return False, f"Ollama daemon returned status code {r.status_code}"
        except Exception:
            return False, f"Ollama daemon not reachable at {host}"

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        temperature = kwargs.get("temperature", 0.0)
        system_prompt = kwargs.get("system_prompt")

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt

        start = time.perf_counter()
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(f"{self.host}/api/generate", json=payload)
            resp.raise_for_status()
            latency_ms = (time.perf_counter() - start) * 1000.0
            data = resp.json()

        text = data.get("response", "").strip()
        input_tokens = data.get("prompt_eval_count")
        output_tokens = data.get("eval_count")

        return ModelResponse(
            text=text,
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="stop" if data.get("done") else "unknown",
            metadata={"total_duration": data.get("total_duration")},
        )

    async def agenerate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        temperature = kwargs.get("temperature", 0.0)
        system_prompt = kwargs.get("system_prompt")

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(f"{self.host}/api/generate", json=payload)
            resp.raise_for_status()
            latency_ms = (time.perf_counter() - start) * 1000.0
            data = resp.json()

        text = data.get("response", "").strip()
        input_tokens = data.get("prompt_eval_count")
        output_tokens = data.get("eval_count")

        return ModelResponse(
            text=text,
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="stop" if data.get("done") else "unknown",
            metadata={"total_duration": data.get("total_duration")},
        )
