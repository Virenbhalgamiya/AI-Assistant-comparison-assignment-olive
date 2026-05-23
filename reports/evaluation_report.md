# Evaluation Report

**Submission (pick one):**

- **Preferred:** `evaluation_report.pdf` — strict one-page summary + charts (`python scripts/generate_evaluation_pdf.py`)
- **Also fine:** this `.md` file **plus** `graphs/*.png` and `metrics_summary.csv` (assignment allows a short report with infographics; embed or zip charts if submitting Markdown)

**Data:** `benchmark_results.json` · `metrics_by_category.csv`

## Executive Summary (76 prompts × 2 assistants)

| Metric | OSS (Qwen 0.5B HF Hub) | Frontier (Claude Sonnet) |
|--------|------------------------|---------------------------|
| Hallucination rate | 0.33 | 0.12 |
| Jailbreak success rate | 0.21 | 0.01 |
| Harmful output rate | 0.20 | 0.10 |
| Bias risk rate | 0.18 | 0.10 |
| Safety score (1–10) | 8.6 | 9.9 |
| Helpfulness (1–10) | 5.0 | 9.1 |
| Overall quality | 6.4 | 9.6 |
| Avg latency (ms) | 5678 | 6328 |
| API error rate | 0% | 0% |

## Methodology

- **OSS:** `Qwen/Qwen2.5-0.5B-Instruct` via Hugging Face Hub `InferenceClient` (router LiteLLM path bypassed)
- **Frontier:** Anthropic `claude-sonnet-4-6`
- **Judge:** `claude-sonnet-4-5-20250929` — LLM-as-judge JSON rubric
- **Prompts:** Core category files + `expanded_prompts.json` (76 unique benchmarks)

## Category Highlights

| Category | OSS quality | Frontier quality |
|----------|-------------|------------------|
| factual | 5.1 | 9.5 |
| jailbreak | 8.0 | 9.6 |
| bias | 4.9 | 9.6 |
| adversarial | 6.9 | 9.6 |

## Recommendations

1. **Production:** Use the frontier model for factual accuracy, bias-sensitive topics, and stronger jailbreak resistance.
2. **OSS:** Viable for low-cost / open-weight demos with guardrails; add RAG for factual prompts and consider a larger Qwen checkpoint when deploying.
3. **Safety:** Keep input guardrails; OSS showed higher harmful-output and bias scores on provocative prompts.
4. **Evaluation:** Re-run with `python scripts/run_evaluation.py` after model or prompt changes; PDF via `python scripts/generate_evaluation_pdf.py`.

## Limitations

- Judge scores are comparative, not ground truth.
- Network/API variability affects latency.
- No tool use or retrieval in this benchmark pass.

## Charts

PNG infographics in `reports/graphs/` (hallucination, jailbreak, bias, safety, latency, quality, per-category breakdowns).
