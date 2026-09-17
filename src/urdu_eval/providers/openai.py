"""OpenAI model provider with lazy imports."""

from __future__ import annotations

import os
import time
from typing import Any

from urdu_eval.models import ModelResponse
from urdu_eval.providers.base import ModelProvider, register_provider


@register_provider("openai")
class OpenAIProvider(ModelProvider):
    """OpenAI API provider for GPT-4o, GPT-4, etc."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.organization = organization

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment variable "
                "or pass api_key to the provider."
            )

        try:
            import openai

            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
            )
            self._async_client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use OpenAIProvider. "
                "Install it using: pip install 'urdu-eval[openai]'"
            ) from exc

    @classmethod
    def check_availability(cls) -> tuple[bool, str | None]:
        try:
            import openai  # noqa: F401

            if not os.getenv("OPENAI_API_KEY"):
                return False, "OPENAI_API_KEY environment variable is not set."
            return True, None
        except ImportError:
            return False, "openai package is not installed."

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1024)
        system_prompt = kwargs.get("system_prompt")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage

        return ModelResponse(
            text=text.strip(),
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason or "stop",
        )

    async def agenerate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1024)
        system_prompt = kwargs.get("system_prompt")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        resp = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage

        return ModelResponse(
            text=text.strip(),
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason or "stop",
        )
