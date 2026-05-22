# AI Assistant Comparison

This project compares two personal assistants:

1. **OSS Assistant** — `Qwen/Qwen2.5-0.5B-Instruct` via Hugging Face Hub inference
2. **Frontier Assistant** — hosted frontier model (default: **Anthropic Claude**; configurable via `.env`)

Both are evaluated on hallucination rate, harmful/bias outputs, jailbreak resistance, refusal handling, latency, and overall quality. A one-page PDF report and charts ship in `reports/`.

## Project Overview

Streamlit app with multi-turn chat, short-term memory, global memory, guardrails, JSONL logging, and an automated benchmark pipeline (76 prompts) scored by an LLM-as-judge.

| Role | Default model | Provider |
|------|---------------|----------|
| OSS | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face Hub |
| Frontier | `claude-sonnet-4-6` | Anthropic (or OpenRouter/OpenAI via LiteLLM) |
| Judge (live re-eval) | configurable | Anthropic, OpenRouter, etc. |

## Features

- Multi-turn chat for OSS and Frontier assistants
- Chat sessions (new / switch / delete) and global memory sidebar
- Safety keyword/regex guardrails and refusal templates
- Evaluation dashboard with bundled results + optional live re-run
- Export: CSV, JSON, PNG charts, PDF report

## Architecture

Shared `BaseAssistant` → provider adapters (`huggingface_hub` for OSS, LiteLLM for frontier/judge). `EvaluationRunner` loads benchmarks from `evals/benchmark_prompts/`, scores responses, writes artifacts under `reports/`. `app/runtime_config.py` handles local vs public HF Space behavior and optional sidebar API-key overrides.

## Folder Structure

```text
ai-assistant-comparison/
├── app/
│   ├── ui.py
│   ├── config.py
│   ├── runtime_config.py
│   ├── memory.py, prompts.py, utils.py, storage.py, providers.py
│   └── assistants/
├── evals/
│   ├── benchmark_prompts/
│   ├── benchmark_loader.py, evaluator.py, metrics.py, judge.py
├── scripts/
│   ├── run_evaluation.py
│   ├── generate_evaluation_pdf.py
│   └── capture_demo_screenshots.py
├── reports/          # PDF, CSV, JSON, graphs, screenshots (committed)
├── tools/            # latency_test.py, expand_prompts.py
├── DEPLOY_HF.md      # Hugging Face Spaces (no Anthropic secret on server)
├── .env.example
├── main.py
└── requirements.txt
```

---

## Local setup (clone and run)

**Prerequisites:** Python 3.11+, Hugging Face account token, frontier API key (Anthropic recommended).

### 1. Clone and install

```bash
git clone <your-repo-url>
cd ai-assistant-comparison
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

Edit `.env` with **your** keys (never commit `.env`).

### 3. Minimum keys for local chat

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `HF_TOKEN` or `OSS_API_KEY` | **Yes** | OSS assistant (Qwen on HF Hub) |
| `FRONTIER_API_KEY` | **Yes** (local) | Frontier assistant (e.g. Claude) |
| `FRONTIER_PROVIDER` | Recommended | `anthropic` for Claude |
| `FRONTIER_MODEL_NAME` | Recommended | e.g. `claude-sonnet-4-6` |
| `JUDGE_API_KEY` | Only for live re-eval | Benchmark judge |
| `PUBLIC_DEMO_MODE` | Local: `false` | See [Public demo vs local](#public-demo-vs-local) |

### 4. Example `.env` (local full stack)

```env
HF_TOKEN=hf_xxxxxxxx
OSS_PROVIDER=huggingface
OSS_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct

PUBLIC_DEMO_MODE=false

FRONTIER_PROVIDER=anthropic
FRONTIER_API_KEY=sk-ant-xxxxxxxx
FRONTIER_MODEL_NAME=claude-sonnet-4-6

JUDGE_PROVIDER=anthropic
JUDGE_API_KEY=sk-ant-xxxxxxxx
JUDGE_MODEL_NAME=claude-sonnet-4-5-20250929
```

**Frontier via OpenRouter** (instead of Anthropic): set `FRONTIER_PROVIDER=openai_compatible`, `FRONTIER_BASE_URL=https://openrouter.ai/api/v1`, and a `FRONTIER_MODEL_NAME` your key supports (e.g. `openai/gpt-4o-mini`).

### 5. Run the app

```bash
streamlit run main.py
```

Open `http://localhost:8501`. Use the sidebar to switch OSS / Frontier, manage chats, and add global memory.

### 6. Sidebar API keys (optional override)

In the UI, **API keys (optional)** fields override `.env` for **this browser session only** (not saved to disk):

- **Anthropic API key** — enables Frontier Assistant without editing `.env`
- **Judge API key** — enables “Run live evaluation suite”

Model names still come from `.env` (`FRONTIER_MODEL_NAME`, `JUDGE_MODEL_NAME`); the UI does not pick models.

---

## View results without API keys

Pre-computed evaluation artifacts are already in the repo:

- `reports/evaluation_report.pdf` — submission comparison (Claude vs Qwen)
- `reports/metrics_summary.csv`, `reports/metrics_by_category.csv`
- `reports/graphs/*.png`

Run Streamlit and open the **Evaluation Dashboard** tab to view bundled tables/charts and download the PDF — no keys required for **viewing** those results.

---

## Re-run evaluation locally

Requires OSS + frontier + judge keys in `.env` (or sidebar judge key for live run only).

```bash
python scripts/run_evaluation.py              # 76 prompts (~30+ min)
python scripts/run_evaluation.py --core-only  # ~22 prompts (faster smoke test)
python scripts/generate_evaluation_pdf.py
```

Outputs: `reports/metrics_summary.csv`, `reports/benchmark_results.json`, `reports/graphs/*.png`, `reports/evaluation_report.pdf`.

---

## Public demo vs local

| | **Local** (`PUBLIC_DEMO_MODE=false`) | **HF Space** (see [DEPLOY_HF.md](DEPLOY_HF.md)) |
|--|--------------------------------------|--------------------------------------------------|
| OSS | Your `HF_TOKEN` in `.env` | Space secret `HF_TOKEN` only |
| Frontier | Your `FRONTIER_API_KEY` in `.env` | Visitor pastes **own** Anthropic key in sidebar (BYOK) |
| App startup | Needs OSS + frontier keys | Needs OSS key only; frontier optional |
| Eval | Re-run with judge key | Bundled PDF/CSV; live re-run needs judge BYOK |

Frontier is **always** a proper hosted model (Claude by default), **not** a second Qwen.

---

## Environment variables (reference)

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face token for OSS (`OSS_PROVIDER=huggingface`) |
| `OSS_API_KEY` | Alternative to `HF_TOKEN` for OSS |
| `OSS_PROVIDER`, `OSS_MODEL_NAME`, `OSS_BASE_URL` | OSS routing |
| `FRONTIER_PROVIDER`, `FRONTIER_API_KEY`, `FRONTIER_BASE_URL`, `FRONTIER_MODEL_NAME` | Frontier routing |
| `JUDGE_PROVIDER`, `JUDGE_API_KEY`, `JUDGE_BASE_URL`, `JUDGE_MODEL_NAME` | Live benchmark judge |
| `PUBLIC_DEMO_MODE` | `true` = HF-style (OSS only required to load) |
| `MEMORY_WINDOW`, `TEMPERATURE`, `MAX_NEW_TOKENS` | Generation settings |
| `DEBUG_LITELLM` | `true` for provider debug logs |

**Supported providers** (via LiteLLM): `huggingface`, `anthropic` / `claude`, `openai_compatible` / `openrouter` / `groq` / `azure` / `anyscale`.

OSS uses `huggingface_hub.InferenceClient` (not the HF router LiteLLM path, which may reject models on some accounts).

---

## Tests and tools

```bash
pytest -q
python tools/latency_test.py
```

## Demo screenshots

See `reports/screenshots/`. Regenerate while Streamlit is running:

```bash
pip install playwright
python -m playwright install chromium
streamlit run main.py
# other terminal:
python scripts/capture_demo_screenshots.py
```

## Evaluation methodology

76 prompts (core JSON files + `expanded_prompts.json`): factual, adversarial, jailbreak, bias. LLM-as-judge JSON rubric → aggregated metrics and charts.

## Tradeoffs

- Hosted inference keeps the project lightweight; no local GPU required for OSS.
- LLM-as-judge scores are comparative, not ground truth.
- Guardrails are lightweight (keyword/regex); production would add output moderation.

## Deployment

### Hugging Face Spaces (recommended public demo)

**[DEPLOY_HF.md](DEPLOY_HF.md)** — only `HF_TOKEN` in Space secrets; Anthropic stays local or BYOK in sidebar.

### Streamlit Cloud

Entry: `main.py`, secrets: same as local `.env` (or OSS-only + BYOK pattern with `PUBLIC_DEMO_MODE=true`).

### Docker

```bash
docker build -f deployment/Dockerfile -t ai-assistant-comparison .
docker run -p 8501:8501 --env-file .env ai-assistant-comparison
```

## Submission artifacts

| Artifact | Path |
|----------|------|
| Source | This repo |
| Evaluation PDF | `reports/evaluation_report.pdf` |
| Demo screenshots | `reports/screenshots/` |
| Optional live demo | HF Space URL |

Email: `work@ollive.ai` with repo link + PDF (+ optional demo link).
