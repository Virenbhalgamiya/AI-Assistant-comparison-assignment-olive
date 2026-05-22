"""Metric aggregation and chart generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.utils import dump_json, ensure_dir, export_dataframe


@dataclass
class MetricSummary:
    assistant: str
    hallucination_rate: float
    harmful_output_rate: float
    jailbreak_success_rate: float
    bias_rate: float
    refusal_rate: float
    refusal_quality: float
    avg_latency_ms: float
    helpfulness: float
    safety: float
    overall_quality: float
    api_error_rate: float


def build_summary(results_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = results_frame.groupby("assistant", dropna=False)
    summary = grouped.agg(
        hallucination_rate=("hallucination_rate", "mean"),
        harmful_output_rate=("harmful_output_rate", "mean"),
        jailbreak_success_rate=("jailbreak_success_rate", "mean"),
        bias_rate=("bias_rate", "mean"),
        refusal_rate=("refusal_rate", "mean"),
        refusal_quality=("refusal_quality", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        helpfulness=("helpfulness", "mean"),
        safety=("safety", "mean"),
        overall_quality=("overall_quality", "mean"),
        api_error_rate=("api_error", "mean"),
    ).reset_index()
    return summary.round(3)


def build_category_summary(results_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = results_frame.groupby(["assistant", "category"], dropna=False)
    summary = grouped.agg(
        hallucination_rate=("hallucination_rate", "mean"),
        harmful_output_rate=("harmful_output_rate", "mean"),
        jailbreak_success_rate=("jailbreak_success_rate", "mean"),
        bias_rate=("bias_rate", "mean"),
        refusal_rate=("refusal_rate", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        helpfulness=("helpfulness", "mean"),
        safety=("safety", "mean"),
        overall_quality=("overall_quality", "mean"),
        n_prompts=("benchmark_id", "count"),
    ).reset_index()
    return summary.round(3)


def export_results(
    results_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    category_frame: pd.DataFrame,
    reports_dir: Path,
) -> None:
    ensure_dir(reports_dir)
    export_dataframe(summary_frame, reports_dir / "metrics_summary.csv")
    export_dataframe(category_frame, reports_dir / "metrics_by_category.csv")
    dump_json(
        {
            "results": results_frame.to_dict(orient="records"),
            "summary": summary_frame.to_dict(orient="records"),
            "by_category": category_frame.to_dict(orient="records"),
        },
        reports_dir / "benchmark_results.json",
    )


def _bar_chart(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    path: Path,
    hue: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if hue and hue in frame.columns:
        pivot = frame.pivot(index=x_col, columns=hue, values=y_col)
        pivot.plot(kind="bar", ax=ax, color=["#0ea5e9", "#f97316"])
    else:
        frame.plot(kind="bar", x=x_col, y=y_col, ax=ax, legend=False, color="#0ea5e9")
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_comparison_charts(
    summary_frame: pd.DataFrame,
    category_frame: pd.DataFrame,
    graphs_dir: Path,
) -> dict[str, Path]:
    ensure_dir(graphs_dir)
    outputs: dict[str, Path] = {}

    overall_specs = [
        ("hallucination_comparison.png", "hallucination_rate", "Hallucination Rate"),
        ("latency_comparison.png", "avg_latency_ms", "Average Latency (ms)"),
        ("safety_comparison.png", "safety", "Safety Score (1-10)"),
        ("refusal_rate_comparison.png", "refusal_rate", "Refusal Rate"),
        ("bias_rate_comparison.png", "bias_rate", "Bias Risk Rate"),
        ("jailbreak_success_comparison.png", "jailbreak_success_rate", "Jailbreak Success Rate"),
        ("helpfulness_comparison.png", "helpfulness", "Helpfulness Score"),
        ("overall_quality_comparison.png", "overall_quality", "Overall Quality"),
    ]

    for file_name, column, title in overall_specs:
        if column not in summary_frame.columns:
            continue
        path = graphs_dir / file_name
        _bar_chart(summary_frame, "assistant", column, title, path)
        outputs[column] = path

    if not category_frame.empty:
        for metric, file_name, title in [
            ("hallucination_rate", "hallucination_by_category.png", "Hallucination by Category"),
            ("jailbreak_success_rate", "jailbreak_by_category.png", "Jailbreak Success by Category"),
            ("bias_rate", "bias_by_category.png", "Bias Risk by Category"),
        ]:
            if metric not in category_frame.columns:
                continue
            path = graphs_dir / file_name
            _bar_chart(category_frame, "category", metric, title, path, hue="assistant")
            outputs[f"category_{metric}"] = path

    return outputs


def frame_from_records(records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)
