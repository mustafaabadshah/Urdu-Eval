"""OpenRouter model provider."""

from __future__ import annotations

import os
from typing import Any

from urdu_eval.providers.base import register_provider
from urdu_eval.providers.http import HTTPProvider


@register_provider("openrouter")
class OpenRouterProvider(HTTPProvider):
    """OpenRouter provider for multi-model API access."""

    name = "openrouter"

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        **kwargs: Any,
    ) -> None:
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key to OpenRouterProvider."
            )
        headers = {
            "HTTP-Referer": "https://github.com/mustafaabadshah/Urdu-Eval",
            "X-Title": "UrduEval Evaluation Toolkit",
        }
        super().__init__(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            model=model,
            api_key=key,
            timeout_seconds=timeout_seconds,
            headers=headers,
            is_openai_compatible=True,
            **kwargs,
        )

    @classmethod
    def check_availability(cls) -> tuple[bool, str | None]:
        if not os.getenv("OPENROUTER_API_KEY"):
            return False, "OPENROUTER_API_KEY environment variable is not set."
        return True, None
