"""Basic tests for provider wiring (non-networking smoke tests)."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path for imports during pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.providers import ProviderFactory, ProviderSettings, _normalize_provider


def test_normalize_provider_aliases():
    assert _normalize_provider("hf") == "huggingface"
    assert _normalize_provider("claude") == "anthropic"
    assert _normalize_provider("openrouter") == "openai"


def test_provider_factory_creates_hf_hub_instance():
    settings = ProviderSettings(provider="huggingface", api_key="test-token", model_name="Qwen/Qwen2.5-0.5B-Instruct")
    prov = ProviderFactory.create(settings)
    assert hasattr(prov, "generate")
    assert prov.__class__.__name__ == "HuggingFaceHubProvider"


def test_provider_factory_requires_hf_token():
    settings = ProviderSettings(provider="huggingface", api_key=None, model_name="Qwen/Qwen2.5-0.5B-Instruct")
    try:
        ProviderFactory.create(settings)
        raised = False
    except ValueError:
        raised = True
    assert raised
