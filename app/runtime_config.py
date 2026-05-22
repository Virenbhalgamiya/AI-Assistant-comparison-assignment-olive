"""Resolve effective config from environment, public demo mode, and optional UI overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.config import AppConfig


@dataclass(frozen=True)
class RuntimeConfig:
    """Effective settings used per request / per Streamlit session."""

    oss_provider: str
    oss_api_key: str | None
    oss_model_name: str
    oss_base_url: str | None
    frontier_provider: str
    frontier_api_key: str | None
    frontier_model_name: str
    frontier_base_url: str | None
    judge_provider: str
    judge_api_key: str | None
    judge_model_name: str
    judge_base_url: str | None
    temperature: float
    max_new_tokens: int
    memory_window: int
    logs_dir: Any
    data_dir: Any
    reports_dir: Any
    graphs_dir: Any
    benchmark_dir: Any
    public_demo_mode: bool

    @property
    def frontier_configured(self) -> bool:
        return bool(self.frontier_api_key)

    @property
    def frontier_display_name(self) -> str:
        return f"Frontier Assistant · {self.frontier_model_name}"


def is_public_demo_mode() -> bool:
    explicit = os.getenv("PUBLIC_DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    on_hf_space = bool(os.getenv("SPACE_ID") or os.getenv("SPACE_TITLE"))
    return explicit or on_hf_space


def resolve_runtime_config(base: AppConfig, session_overrides: dict[str, str] | None = None) -> RuntimeConfig:
    session_overrides = session_overrides or {}
    public_demo = is_public_demo_mode()

    oss_key = base.oss_api_key or base.hf_token
    frontier_key = session_overrides.get("frontier_api_key") or base.frontier_api_key
    judge_key = session_overrides.get("judge_api_key") or base.judge_api_key

    frontier_provider = base.frontier_provider
    frontier_model = base.frontier_model_name
    frontier_base_url = base.frontier_base_url

    if frontier_key and (
        frontier_provider in {"anthropic", "claude"}
        or str(frontier_key).startswith("sk-ant")
    ):
        frontier_provider = "anthropic"

    return RuntimeConfig(
        oss_provider=base.oss_provider,
        oss_api_key=oss_key,
        oss_model_name=base.oss_model_name,
        oss_base_url=base.oss_base_url,
        frontier_provider=frontier_provider,
        frontier_api_key=frontier_key,
        frontier_model_name=frontier_model,
        frontier_base_url=frontier_base_url,
        judge_provider=base.judge_provider,
        judge_api_key=judge_key,
        judge_model_name=base.judge_model_name,
        judge_base_url=base.judge_base_url,
        temperature=base.temperature,
        max_new_tokens=base.max_new_tokens,
        memory_window=base.memory_window,
        logs_dir=base.logs_dir,
        data_dir=base.data_dir,
        reports_dir=base.reports_dir,
        graphs_dir=base.graphs_dir,
        benchmark_dir=base.benchmark_dir,
        public_demo_mode=public_demo,
    )


def missing_keys_for_chat(runtime: RuntimeConfig) -> list[str]:
    """Only OSS is required to load the app; frontier needs Anthropic separately."""
    missing: list[str] = []
    if runtime.oss_provider in {"huggingface", "hf"} and not runtime.oss_api_key:
        missing.append("HF_TOKEN or OSS_API_KEY")
    elif not runtime.oss_api_key:
        missing.append("OSS_API_KEY")
    if not runtime.public_demo_mode and not runtime.frontier_configured:
        if runtime.frontier_provider in {"anthropic", "claude", "openai", "openai_compatible", "openrouter", "groq", "azure", "anyscale"}:
            missing.append("FRONTIER_API_KEY")
    return missing


def missing_keys_for_eval(runtime: RuntimeConfig) -> list[str]:
    missing = list(missing_keys_for_chat(runtime))
    if runtime.judge_provider in {"anthropic", "claude"} and not runtime.judge_api_key:
        missing.append("JUDGE_API_KEY (optional — bundled report is shown without it)")
    if runtime.judge_provider in {"openai", "openai_compatible", "openrouter", "groq", "azure", "anyscale"} and not runtime.judge_api_key:
        missing.append("JUDGE_API_KEY")
    return missing
