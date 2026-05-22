"""Expand and split benchmark prompts into per-category files.

Usage: python tools/expand_prompts.py
This script loads `evals/benchmark_prompts/expanded_prompts.json`, applies
simple paraphrase templates to reach ~500 prompts, and writes per-category
files under `evals/benchmark_prompts/`.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "evals" / "benchmark_prompts"
INPUT_FILE = PROMPT_DIR / "expanded_prompts.json"


def load_seed():
    with INPUT_FILE.open("r", encoding="utf8") as f:
        return json.load(f)


PARAPHRASE_TEMPLATES = [
    "In brief, {prompt}",
    "Please answer: {prompt}",
    "Short answer: {prompt}",
    "Respond concisely: {prompt}",
    "Explain why: {prompt}",
]


def expand(prompts, target=500):
    out = []
    for p in prompts:
        out.append(p)
    i = 0
    while len(out) < target:
        base = random.choice(prompts)
        template = random.choice(PARAPHRASE_TEMPLATES)
        new_prompt_text = template.format(prompt=base["prompt"])
        new_id = f"gen-{i:04d}-{base['id']}"
        out.append({
            "id": new_id,
            "category": base["category"],
            "prompt": new_prompt_text,
            "expected": base.get("expected", "")
        })
        i += 1
    return out


def split_and_write(prompts):
    by_cat = {}
    for p in prompts:
        by_cat.setdefault(p["category"], []).append(p)

    for cat, items in by_cat.items():
        out_file = PROMPT_DIR / f"{cat}.json"
        with out_file.open("w", encoding="utf8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(items)} prompts to {out_file}")


def main():
    seed = load_seed()
    expanded = expand(seed, target=500)
    split_and_write(expanded)


if __name__ == "__main__":
    main()
