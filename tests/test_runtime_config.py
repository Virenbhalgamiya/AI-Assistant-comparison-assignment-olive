"""Tests for public demo runtime resolution."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import AppConfig
from app.runtime_config import missing_keys_for_chat, resolve_runtime_config


def _base_config(**kwargs) -> AppConfig:
    defaults = dict(
        hf_token="hf_test",
        oss_api_key="hf_test",
        oss_provider="huggingface",
        oss_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        oss_base_url=None,
        frontier_provider="anthropic",
        frontier_api_key=None,
        frontier_model_name="claude-sonnet-4-6",
        frontier_base_url=None,
        judge_provider="anthropic",
        judge_api_key=None,
        judge_model_name="claude-sonnet-4-5",
        judge_base_url=None,
        debug_litellm=False,
        memory_window=8,
        temperature=0.4,
        max_new_tokens=512,
        logs_dir=ROOT / "logs",
        data_dir=ROOT / "data",
        reports_dir=ROOT / "reports",
        graphs_dir=ROOT / "reports" / "graphs",
        benchmark_dir=ROOT / "evals" / "benchmark_prompts",
    )
    defaults.update(kwargs)
    return AppConfig(**defaults)


def test_public_demo_does_not_use_qwen_frontier_fallback(monkeypatch):
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "true")
    runtime = resolve_runtime_config(_base_config())
    assert runtime.frontier_model_name == "claude-sonnet-4-6"
    assert runtime.frontier_provider == "anthropic"
    assert runtime.frontier_configured is False
    assert missing_keys_for_chat(runtime) == []


def test_session_anthropic_key_enables_frontier(monkeypatch):
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "true")
    runtime = resolve_runtime_config(_base_config(), {"frontier_api_key": "sk-ant-test"})
    assert runtime.frontier_configured is True
    assert runtime.frontier_provider == "anthropic"
