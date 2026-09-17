"""Tests for model provider abstractions and implementations."""

import pytest

from urdu_eval.providers.base import get_provider, list_providers
from urdu_eval.providers.http import HTTPProvider
from urdu_eval.providers.mock import MockProvider
from urdu_eval.providers.openai import OpenAIProvider


def test_mock_provider_generate_sync() -> None:
    """Test MockProvider synchronous generation."""
    provider = MockProvider(model="test-mock")
    resp = provider.generate("پاکستان کا دارالحکومت کیا ہے؟")
    assert resp.model == "test-mock"
    assert resp.provider == "mock"
    assert resp.text == "اسلام آباد"
    assert resp.latency_ms >= 0.0
    assert resp.input_tokens is not None
    assert resp.output_tokens is not None
    assert resp.finish_reason == "stop"


@pytest.mark.asyncio
async def test_mock_provider_generate_async() -> None:
    """Test MockProvider asynchronous generation."""
    provider = MockProvider(model="test-mock-async")
    resp = await provider.agenerate("dar-ul-hukoomat")
    assert resp.text == "Islamabad"
    assert resp.provider == "mock"


def test_mock_provider_fallback_and_failure() -> None:
    """Test MockProvider fallback response and simulated failure."""
    provider = MockProvider(default_response="عام جواب")
    resp = provider.generate("کوئی غیر متعلقہ سوال")
    assert resp.text == "عام جواب"

    failing_provider = MockProvider(should_fail=True, failure_message="Test API Error")
    with pytest.raises(RuntimeError, match="Test API Error"):
        failing_provider.generate("کچھ بھی")


def test_provider_registry() -> None:
    """Test get_provider registry resolution and error handling."""
    mock_p = get_provider("mock")
    assert isinstance(mock_p, MockProvider)

    with pytest.raises(ValueError, match="is not registered"):
        get_provider("non_existent_provider_xyz")


def test_list_providers() -> None:
    """Test list_providers returns registered providers and availability."""
    all_providers = list_providers()
    assert "mock" in all_providers
    assert all_providers["mock"]["available"] is True
    assert "http" in all_providers
    assert "openai" in all_providers


def test_lazy_imports_and_key_validation() -> None:
    """Test that cloud providers raise clear ValueError when API keys are missing."""
    with pytest.raises(ValueError, match="OpenAI API key not found"):
        OpenAIProvider(api_key=None)


def test_http_provider_payload_and_parsing() -> None:
    """Test HTTPProvider payload creation and response parsing."""
    http_p = HTTPProvider(
        endpoint="http://localhost:8000/v1/chat/completions",
        model="custom-llama",
        is_openai_compatible=True,
    )
    payload = http_p._prepare_payload("سلام", temperature=0.2, max_tokens=100)
    assert payload["model"] == "custom-llama"
    assert payload["messages"][0]["content"] == "سلام"
    assert payload["temperature"] == 0.2

    # Test choice format parsing
    mock_api_data = {
        "choices": [
            {
                "message": {"content": "وعلیکم السلام"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 8},
    }
    resp = http_p._parse_response(mock_api_data, latency_ms=150.0)
    assert resp.text == "وعلیکم السلام"
    assert resp.latency_ms == 150.0
    assert resp.input_tokens == 5
    assert resp.output_tokens == 8

    # Test ollama response format parsing
    ollama_data = {
        "response": "جواب",
        "done": True,
        "prompt_eval_count": 4,
        "eval_count": 2,
    }
    resp_ollama = http_p._parse_response(ollama_data, latency_ms=80.0)
    assert resp_ollama.text == "جواب"
    assert resp_ollama.input_tokens == 4
    assert resp_ollama.output_tokens == 2
