"""Run the full benchmark suite from the command line.

Usage:
  python scripts/run_evaluation.py
  python scripts/run_evaluation.py --core-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.assistants.frontier_assistant import FrontierAssistant
from app.assistants.oss_assistant import OSSAssistant
from app.config import load_config
from app.memory import MemoryManager
from app.prompts import compose_system_prompt
from evals.evaluator import EvaluationRunner
from evals.judge import HostedJudge


def build_assistant(name: str, config):
    memory = MemoryManager(window_size=config.memory_window)
    system_context = f"[ASSISTANT_ROLE:{name}]\n"
    if name == "OSS Assistant":
        return OSSAssistant(
            provider=config.oss_provider,
            api_key=config.oss_api_key,
            memory=memory,
            system_context=system_context,
            model_name=config.oss_model_name,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            base_url=config.oss_base_url,
        )
    return FrontierAssistant(
        provider=config.frontier_provider,
        api_key=config.frontier_api_key,
        memory=memory,
        system_context=system_context,
        model_name=config.frontier_model_name,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        base_url=config.frontier_base_url,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run assistant benchmark evaluation.")
    parser.add_argument("--core-only", action="store_true", help="Skip expanded_prompts.json")
    args = parser.parse_args()

    config = load_config()
    missing = config.missing_keys()
    if missing:
        print("Missing environment variables:", ", ".join(missing))
        sys.exit(1)

    judge = HostedJudge(
        provider=config.judge_provider,
        api_key=config.judge_api_key,
        model_name=config.judge_model_name,
        base_url=config.judge_base_url,
    )
    runner = EvaluationRunner(judge, config.reports_dir, config.graphs_dir)
    assistants = {
        "OSS Assistant": lambda: build_assistant("OSS Assistant", config),
        "Frontier Assistant": lambda: build_assistant("Frontier Assistant", config),
    }

    def log(msg: str) -> None:
        print(msg, flush=True)

    log("Starting benchmark evaluation...")
    log(f"  OSS model: {config.oss_model_name}")
    log(f"  Frontier model: {config.frontier_model_name}")
    log(f"  Judge model: {config.judge_model_name}")

    results, summary, by_category, charts = runner.run(
        assistants,
        config.benchmark_dir,
        include_expanded=not args.core_only,
        progress_callback=log,
    )

    log("\n=== Summary ===")
    log(summary.to_string(index=False))
    log("\n=== By category (sample) ===")
    log(by_category.head(12).to_string(index=False))
    log(f"\nArtifacts written to {config.reports_dir}")
    for label, path in charts.items():
        log(f"  chart: {path}")


if __name__ == "__main__":
    main()
