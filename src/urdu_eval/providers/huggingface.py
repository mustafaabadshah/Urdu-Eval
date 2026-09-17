"""HuggingFace transformers model provider with lazy imports."""

from __future__ import annotations

import os
import time
from typing import Any

from urdu_eval.models import ModelResponse
from urdu_eval.providers.base import ModelProvider, register_provider


@register_provider("huggingface")
@register_provider("hf")
class HuggingFaceProvider(ModelProvider):
    """Local HuggingFace pipeline provider."""

    name = "huggingface"

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str | None = None,
        token: str | None = None,
        torch_dtype: str = "auto",
        **kwargs: Any,
    ) -> None:
        self.model_name = model
        self.device = device
        self.token = token or os.getenv("HF_TOKEN")
        self.torch_dtype = torch_dtype

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.token,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.token,
                torch_dtype=torch_dtype,
                device_map="auto" if device is None else device,
                trust_remote_code=True,
            )
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' and 'torch' packages are required for HuggingFaceProvider. "
                "Install with: pip install 'urdu-eval[hf]'"
            ) from exc

    @classmethod
    def check_availability(cls) -> tuple[bool, str | None]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True, None
        except ImportError:
            return False, "transformers/torch packages are not installed."

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 512)

        start = time.perf_counter()
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        generated_text = outputs[0]["generated_text"]
        # If output includes prompt, strip prompt prefix
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :].strip()

        input_tokens = len(self._tokenizer.encode(prompt))
        output_tokens = len(self._tokenizer.encode(generated_text))

        return ModelResponse(
            text=generated_text.strip(),
            model=self.model_name,
            provider=self.name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="stop",
        )

    async def agenerate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        # In-memory local inference executed in sync worker
        return self.generate(prompt, **kwargs)
