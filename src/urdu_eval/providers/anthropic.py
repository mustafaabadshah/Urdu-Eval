"""Anthropic Claude model provider with lazy imports."""

from __future__ import annotations

import os
import time
from typing import Any

from urdu_eval.models import ModelResponse
from urdu_eval.providers.base import ModelProvider, register_provider


@register_provider("anthropic")
class AnthropicProvider(ModelProvider):
    """Anthropic API provider for Claude-3.5, Claude-3, etc."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set the ANTHROPIC_API_KEY environment variable "
                "or pass api_key to the provider."
            )

        try:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self._async_client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required to use AnthropicProvider. "
                "Install it using: pip install 'urdu-eval[anthropic]'"
            ) from exc

    @classmethod
    def check_availability(cls) -> tuple[bool, str | None]:
        try:
            import anthropic  # noqa: F401

            if not os.getenv("ANTHROPIC_API_KEY"):
                return False, "ANTHROPIC_API_KEY environment variable is not set."
            return True, None
        except ImportError:
            return False, "anthropic package is not installed."

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1024)
        system_prompt = kwargs.get("system_prompt")

        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt

        start = time.perf_counter()
        resp = self._client.messages.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text += block.text

        usage = resp.usage
        input_tokens = usage.input_tokens if usage else None
        output_tokens = usage.output_tokens if usage else None

        return ModelResponse(
            text=text.strip(),
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=resp.stop_reason or "stop",
        )

    async def agenerate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1024)
        system_prompt = kwargs.get("system_prompt")

        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt

        start = time.perf_counter()
        resp = await self._async_client.messages.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text += block.text

        usage = resp.usage
        input_tokens = usage.input_tokens if usage else None
        output_tokens = usage.output_tokens if usage else None

        return ModelResponse(
            text=text.strip(),
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=resp.stop_reason or "stop",
        )
