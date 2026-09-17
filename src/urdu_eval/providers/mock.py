"""Mock model provider for offline testing and continuous integration."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from urdu_eval.models import ModelResponse
from urdu_eval.providers.base import ModelProvider, register_provider


@register_provider("mock")
class MockProvider(ModelProvider):
    """Deterministic Mock Provider for testing and CI without external API calls."""

    name = "mock"

    def __init__(
        self,
        model: str = "mock-urdu-model",
        default_response: str = "اسلام آباد",
        canned_responses: dict[str, str] | None = None,
        simulate_latency_ms: float = 10.0,
        should_fail: bool = False,
        failure_message: str = "Mock provider simulated error",
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.default_response = default_response
        self.canned_responses = (
            canned_responses
            if canned_responses is not None
            else {
                "دارالحکومت": "اسلام آباد",
                "dar-ul-hukoomat": "Islamabad",
                "darul hukoomat": "Islamabad",
                "Faisal Masjid": "Islamabad",
                "سیب": "۳",
                "Translate to English": "Seeking knowledge is an obligation upon every Muslim man and woman.",
            }
        )
        self.simulate_latency_ms = simulate_latency_ms
        self.should_fail = should_fail
        self.failure_message = failure_message

    @classmethod
    def check_availability(cls) -> tuple[bool, str | None]:
        """Mock provider is always available."""
        return True, None

    def _resolve_text(self, prompt: str) -> str:
        for keyword, response in self.canned_responses.items():
            if keyword in prompt:
                return response
        return self.default_response

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate response synchronously."""
        if self.should_fail:
            raise RuntimeError(self.failure_message)

        start = time.perf_counter()
        if self.simulate_latency_ms > 0:
            time.sleep(self.simulate_latency_ms / 1000.0)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = self._resolve_text(prompt)
        input_tokens = len(prompt.split())
        output_tokens = len(text.split())

        return ModelResponse(
            text=text,
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="stop",
            metadata={"simulated": True},
        )

    async def agenerate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate response asynchronously."""
        if self.should_fail:
            raise RuntimeError(self.failure_message)

        start = time.perf_counter()
        if self.simulate_latency_ms > 0:
            await asyncio.sleep(self.simulate_latency_ms / 1000.0)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = self._resolve_text(prompt)
        input_tokens = len(prompt.split())
        output_tokens = len(text.split())

        return ModelResponse(
            text=text,
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="stop",
            metadata={"simulated": True},
        )
