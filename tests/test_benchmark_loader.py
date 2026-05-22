"""Tests for benchmark loading."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.benchmark_loader import load_all_benchmarks, normalize_category


def test_normalize_category_aliases():
    assert normalize_category("hallucination") == "factual"
    assert normalize_category("safety") == "adversarial"


def test_load_all_benchmarks_includes_core_and_expanded():
    benchmark_dir = ROOT / "evals" / "benchmark_prompts"
    prompts = load_all_benchmarks(benchmark_dir, include_expanded=True)
    assert len(prompts) >= 22
    ids = {p.id for p in prompts}
    assert "factual-1" in ids
    assert "halluc-001" in ids
