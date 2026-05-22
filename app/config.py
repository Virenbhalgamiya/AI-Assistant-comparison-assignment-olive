"""Configuration helpers for the assistant comparison project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    hf_token: str | None
    oss_api_key: str | None
    oss_provider: str = "huggingface"
    oss_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    oss_base_url: str | None = None
    frontier_provider: str = "openai_compatible"
    frontier_api_key: str | None = None
    frontier_model_name: str = "deepseek/deepseek-v4-flash:free"
    frontier_base_url: str = "https://openrouter.ai/api/v1"
    judge_provider: str = "openai_compatible"
    judge_api_key: str | None = None
    judge_model_name: str = "google/gemma-4-26b-a4b-it:free"
    judge_base_url: str = "https://openrouter.ai/api/v1"
    debug_litellm: bool = False
    memory_window: int = 8
    temperature: float = 0.4
    max_new_tokens: int = 512
    logs_dir: Path = ROOT_DIR / "logs"
    data_dir: Path = ROOT_DIR / "data"
    reports_dir: Path = ROOT_DIR / "reports"
    graphs_dir: Path = ROOT_DIR / "reports" / "graphs"
    benchmark_dir: Path = ROOT_DIR / "evals" / "benchmark_prompts"

    def missing_keys(self) -> list[str]:
        missing: list[str] = []
        if self.oss_provider in {"huggingface", "hf"} and not (self.hf_token or self.oss_api_key):
            missing.append("HF_TOKEN or OSS_API_KEY")
        if self.oss_provider not in {"huggingface", "hf"} and not self.oss_api_key:
            missing.append("OSS_API_KEY")
        if self.frontier_provider in {"openai", "openai_compatible", "openrouter", "groq", "azure", "anyscale"} and not self.frontier_api_key:
            missing.append("FRONTIER_API_KEY")
        if self.frontier_provider in {"anthropic", "claude"} and not self.frontier_api_key:
            missing.append("FRONTIER_API_KEY")
        if self.judge_provider in {"openai", "openai_compatible", "openrouter", "groq", "azure", "anyscale"} and not self.judge_api_key:
            missing.append("JUDGE_API_KEY")
        if self.judge_provider in {"anthropic", "claude"} and not self.judge_api_key:
            missing.append("JUDGE_API_KEY")
        return missing


def load_config() -> AppConfig:
    """Load environment variables from .env and return a typed config."""

    load_dotenv(ROOT_DIR / ".env")
    return AppConfig(
        hf_token=os.getenv("HF_TOKEN"),
        oss_api_key=os.getenv("OSS_API_KEY", os.getenv("HF_TOKEN")),
        oss_provider=os.getenv("OSS_PROVIDER", "huggingface"),
        oss_model_name=os.getenv("OSS_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        oss_base_url=os.getenv("OSS_BASE_URL"),
        frontier_provider=os.getenv("FRONTIER_PROVIDER", "openai_compatible"),
        frontier_api_key=os.getenv("FRONTIER_API_KEY"),
        frontier_model_name=os.getenv("FRONTIER_MODEL_NAME", "deepseek/deepseek-v4-flash:free"),
        frontier_base_url=os.getenv("FRONTIER_BASE_URL"),
        judge_provider=os.getenv("JUDGE_PROVIDER", "openai_compatible"),
        judge_api_key=os.getenv("JUDGE_API_KEY"),
        judge_model_name=os.getenv("JUDGE_MODEL_NAME", "google/gemma-4-26b-a4b-it:free"),
        judge_base_url=os.getenv("JUDGE_BASE_URL"),
        debug_litellm=os.getenv("DEBUG_LITELLM", "false").strip().lower() in {"1", "true", "yes", "on"},
        memory_window=int(os.getenv("MEMORY_WINDOW", "8")),
        temperature=float(os.getenv("TEMPERATURE", "0.4")),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "512")),
    )
