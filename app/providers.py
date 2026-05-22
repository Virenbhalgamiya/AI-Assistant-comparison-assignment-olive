"""LiteLLM-based provider adapter and Hugging Face Hub inference for OSS models."""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Protocol

import litellm
from litellm import completion


def _enable_debug_if_requested() -> None:
    debug_enabled = os.getenv("DEBUG_LITELLM", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not debug_enabled:
        return
    turn_on_debug = getattr(litellm, "_turn_on_debug", None)
    if callable(turn_on_debug):
        turn_on_debug()
    if hasattr(litellm, "set_verbose"):
        try:
            litellm.set_verbose = True
        except Exception:
            pass


_enable_debug_if_requested()


class TextProvider(Protocol):
    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> tuple[str, dict[str, object]]:
        raise NotImplementedError


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    api_key: str | None
    model_name: str
    base_url: str | None = None


class HuggingFaceHubProvider:
    """Use huggingface_hub InferenceClient (works when HF router rejects the model)."""

    def __init__(self, settings: ProviderSettings) -> None:
        from huggingface_hub import InferenceClient

        if not settings.api_key:
            raise ValueError("HF_TOKEN or OSS_API_KEY is required for Hugging Face inference")
        self.model_name = settings.model_name
        self.client = InferenceClient(token=settings.api_key, base_url=settings.base_url)

    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> tuple[str, dict[str, object]]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.chat_completion(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                choice = response.choices[0]
                content = getattr(choice.message, "content", None) or ""
                return content, {"provider": "huggingface_hub", "model": self.model_name}
            except Exception as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"HuggingFace Hub call failed: {last_error}") from last_error


class LiteLLMProvider:
    def __init__(self, settings: ProviderSettings) -> None:
        self.provider = _normalize_provider(settings.provider)
        self.api_key = settings.api_key
        self.model_name = settings.model_name
        self.base_url = settings.base_url

    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> tuple[str, dict[str, object]]:
        _enable_debug_if_requested()
        kwargs: dict[str, object] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "custom_llm_provider": self.provider,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = completion(**kwargs)
                return _extract_text(response), {"provider": self.provider, "model": self.model_name}
            except Exception as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        if os.getenv("DEBUG_LITELLM", "false").strip().lower() in {"1", "true", "yes", "on"}:
            print(f"[LiteLLM debug] provider={self.provider} model={self.model_name}", file=sys.stderr)
            traceback.print_exc()
        raise RuntimeError(
            f"LiteLLM call failed for provider={self.provider} model={self.model_name}: {last_error}"
        ) from last_error


class ProviderFactory:
    @staticmethod
    def create(settings: ProviderSettings) -> TextProvider:
        provider = settings.provider.strip().lower()
        if provider in {"", "none"}:
            raise ValueError("A provider name is required")
        if provider in {"huggingface", "hf", "huggingface_hub"}:
            return HuggingFaceHubProvider(settings)
        return LiteLLMProvider(settings)


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    aliases = {
        "hf": "huggingface",
        "huggingface_hub": "huggingface",
        "claude": "anthropic",
        "openrouter": "openai",
        "openai_compatible": "openai",
    }
    return aliases.get(normalized, normalized)


def _extract_text(completion_response: Any) -> str:
    if hasattr(completion_response, "choices") and completion_response.choices:
        choice = completion_response.choices[0]
        message = getattr(choice, "message", None)
        if message and hasattr(message, "content"):
            return message.content or ""
        if hasattr(choice, "text"):
            return choice.text or ""
    if isinstance(completion_response, dict):
        choices = completion_response.get("choices", [])
        if choices:
            choice = choices[0]
            if isinstance(choice, dict):
                return choice.get("message", {}).get("content") or choice.get("text", "")
    return str(completion_response)
