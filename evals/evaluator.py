"""Benchmark runner for the two assistants."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.assistants.base_assistant import AssistantResponse
from evals.benchmark_loader import BenchmarkPrompt, load_all_benchmarks
from evals.judge import HostedJudge
from evals.metrics import (
    build_category_summary,
    build_summary,
    export_results,
    frame_from_records,
    render_comparison_charts,
)


class EvaluationRunner:
    def __init__(self, judge: HostedJudge, reports_dir: Path, graphs_dir: Path) -> None:
        self.judge = judge
        self.reports_dir = reports_dir
        self.graphs_dir = graphs_dir

    def run(
        self,
        assistants: dict[str, Callable[[], object]],
        benchmark_dir: Path,
        include_expanded: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
        prompts = load_all_benchmarks(benchmark_dir, include_expanded=include_expanded)
        total = len(prompts) * len(assistants)
        records: list[dict[str, object]] = []
        step = 0

        for item in prompts:
            for assistant_name, assistant_factory in assistants.items():
                step += 1
                if progress_callback:
                    progress_callback(f"[{step}/{total}] {assistant_name} · {item.category} · {item.id}")

                assistant = assistant_factory()
                result: AssistantResponse = assistant.chat(item.prompt)
                judge_score = self.judge.score(item.prompt, result.content, item.category)

                hallucination_rate = round(max(0.0, min(1.0, judge_score.hallucination_risk / 10.0)), 3)
                harmful_output_rate = round(max(0.0, min(1.0, judge_score.harmfulness / 10.0)), 3)
                jailbreak_success_rate = round(max(0.0, min(1.0, (10.0 - judge_score.jailbreak_resistance) / 10.0)), 3)
                bias_rate = round(max(0.0, min(1.0, judge_score.bias_risk / 10.0)), 3)
                refusal_rate = 1.0 if result.refusal else 0.0
                api_error = 1.0 if "api error" in result.content.lower() else 0.0

                records.append(
                    {
                        "assistant": assistant_name,
                        "assistant_model": result.model_name,
                        "benchmark_id": item.id,
                        "category": item.category,
                        "prompt": item.prompt,
                        "expected": item.expected,
                        "response": result.content,
                        "latency_ms": result.latency_ms,
                        "refusal_rate": refusal_rate,
                        "hallucination_rate": hallucination_rate,
                        "harmful_output_rate": harmful_output_rate,
                        "jailbreak_success_rate": jailbreak_success_rate,
                        "bias_rate": bias_rate,
                        "refusal_quality": judge_score.refusal_quality,
                        "helpfulness": judge_score.helpfulness,
                        "safety": judge_score.safety,
                        "overall_quality": judge_score.overall_quality,
                        "factuality": judge_score.factuality,
                        "api_error": api_error,
                        "guardrail_blocked": 1.0 if result.safety_triggered else 0.0,
                        "rationale": judge_score.rationale,
                    }
                )

        results_frame = frame_from_records(records)
        summary_frame = build_summary(results_frame)
        category_frame = build_category_summary(results_frame)
        charts = render_comparison_charts(summary_frame, category_frame, self.graphs_dir)
        export_results(results_frame, summary_frame, category_frame, self.reports_dir)
        return results_frame, summary_frame, category_frame, charts
