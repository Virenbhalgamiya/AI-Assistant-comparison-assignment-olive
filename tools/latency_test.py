"""Simple latency test runner for providers.

Usage: python tools/latency_test.py
It will load `app.config.load_config()` and attempt a short generate for configured providers.
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

# Make project root importable when running this script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.providers import ProviderFactory, ProviderSettings


def time_provider_run(settings: ProviderSettings, prompt: list[dict[str, str]]):
    prov = ProviderFactory.create(settings)
    t0 = time.perf_counter()
    try:
        text, meta = prov.generate(prompt, max_tokens=64, temperature=0.0)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"Provider {settings.provider} model={settings.model_name} latency_ms={elapsed:.1f}")
        print("Sample response:", (text or '')[:200])
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"Provider {settings.provider} error after {elapsed:.1f}ms: {exc}")


def main():
    cfg = load_config()
    prompt = [{"role": "user", "content": "Say hello in one sentence."}]

    oss_settings = ProviderSettings(provider=cfg.oss_provider, api_key=cfg.oss_api_key or cfg.hf_token, model_name=cfg.oss_model_name, base_url=cfg.oss_base_url)
    frontier_settings = ProviderSettings(provider=cfg.frontier_provider, api_key=cfg.frontier_api_key, model_name=cfg.frontier_model_name, base_url=cfg.frontier_base_url)

    print("Testing OSS provider...")
    time_provider_run(oss_settings, prompt)
    print("Testing Frontier provider...")
    time_provider_run(frontier_settings, prompt)


if __name__ == "__main__":
    main()
