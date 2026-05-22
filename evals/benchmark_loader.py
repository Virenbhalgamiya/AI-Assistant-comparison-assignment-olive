"""Load and merge benchmark prompt sets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkPrompt:
    id: str
    category: str
    prompt: str
    expected: str | None = None


_CATEGORY_ALIASES = {
    "hallucination": "factual",
    "safety": "adversarial",
    "helpfulness": "factual",
}


def normalize_category(category: str) -> str:
    return _CATEGORY_ALIASES.get(category.strip().lower(), category.strip().lower())


def load_benchmark_file(path: Path, default_category: str | None = None) -> list[BenchmarkPrompt]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    prompts: list[BenchmarkPrompt] = []
    for item in payload:
        category = normalize_category(item.get("category") or default_category or "general")
        prompts.append(
            BenchmarkPrompt(
                id=str(item["id"]),
                category=category,
                prompt=str(item["prompt"]),
                expected=item.get("expected"),
            )
        )
    return prompts


def load_all_benchmarks(benchmark_dir: Path, include_expanded: bool = True) -> list[BenchmarkPrompt]:
    """Load category files and optional expanded set, deduplicated by prompt id."""
    files = [
        ("factual.json", "factual"),
        ("jailbreak.json", "jailbreak"),
        ("bias.json", "bias"),
        ("adversarial.json", "adversarial"),
    ]
    merged: list[BenchmarkPrompt] = []
    seen_ids: set[str] = set()

    for file_name, category in files:
        path = benchmark_dir / file_name
        if path.exists():
            for item in load_benchmark_file(path, category):
                if item.id not in seen_ids:
                    merged.append(item)
                    seen_ids.add(item.id)

    expanded_path = benchmark_dir / "expanded_prompts.json"
    if include_expanded and expanded_path.exists():
        for item in load_benchmark_file(expanded_path):
            if item.id not in seen_ids:
                merged.append(item)
                seen_ids.add(item.id)

    return merged
