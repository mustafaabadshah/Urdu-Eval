"""Base protocol and registry for model providers."""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from urdu_eval.models import ModelResponse


@runtime_checkable
class ModelProvider(Protocol):
    """Protocol that all model providers must implement."""

    name: str

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response synchronously for a single prompt."""
        ...

    async def agenerate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response asynchronously for a single prompt."""
        ...


_PROVIDER_REGISTRY: dict[str, type[ModelProvider]] = {}


def register_provider(name: str):
    """Decorator to register a provider class."""

    def decorator(cls: type[ModelProvider]):
        _PROVIDER_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_provider(name: str, **kwargs: Any) -> ModelProvider:
    """Instantiate a provider by name."""
    provider_key = name.lower()

    # Trigger dynamic discovery if not yet in registry
    if provider_key not in _PROVIDER_REGISTRY:
        _discover_builtin_providers()

    if provider_key not in _PROVIDER_REGISTRY:
        available = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Provider '{name}' is not registered. Available providers: {available}")

    provider_cls = _PROVIDER_REGISTRY[provider_key]
    return provider_cls(**kwargs)


def list_providers() -> dict[str, dict[str, Any]]:
    """List all registered providers and their availability status."""
    _discover_builtin_providers()

    results: dict[str, dict[str, Any]] = {}
    for name, cls in _PROVIDER_REGISTRY.items():
        is_available = True
        error_message = None
        if hasattr(cls, "check_availability"):
            is_available, error_message = cls.check_availability()

        results[name] = {
            "name": name,
            "class": cls.__name__,
            "available": is_available,
            "error": error_message,
        }
    return results


def _discover_builtin_providers() -> None:
    """Import built-in provider modules to register them."""
    # Mock
    try:
        from urdu_eval.providers.mock import MockProvider  # noqa: F401
    except ImportError:
        pass

    # HTTP
    try:
        from urdu_eval.providers.http import HTTPProvider  # noqa: F401
    except ImportError:
        pass

    # OpenAI
    try:
        from urdu_eval.providers.openai import OpenAIProvider  # noqa: F401
    except ImportError:
        pass

    # Anthropic
    try:
        from urdu_eval.providers.anthropic import AnthropicProvider  # noqa: F401
    except ImportError:
        pass

    # Ollama
    try:
        from urdu_eval.providers.ollama import OllamaProvider  # noqa: F401
    except ImportError:
        pass

    # OpenRouter
    try:
        from urdu_eval.providers.openrouter import OpenRouterProvider  # noqa: F401
    except ImportError:
        pass

    # HuggingFace
    try:
        from urdu_eval.providers.huggingface import HuggingFaceProvider  # noqa: F401
    except ImportError:
        pass


class BaseTimer:
    """Context manager to measure execution latency."""

    def __enter__(self) -> BaseTimer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end = time.perf_counter()
        self.latency_ms = (self.end - self.start) * 1000.0
