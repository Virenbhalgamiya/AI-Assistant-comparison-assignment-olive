# Deploy on Hugging Face Spaces (no Anthropic secret on server)

Public Space setup:

| Assistant | Model | Keys on server |
|-----------|--------|----------------|
| **OSS** | `Qwen/Qwen2.5-0.5B-Instruct` (HF Hub) | `HF_TOKEN` only |
| **Frontier** | `claude-sonnet-4-6` (Anthropic) | **Not stored** — visitor pastes key in sidebar (BYOK) |

The **benchmark report** in `reports/evaluation_report.pdf` was produced locally with Claude vs Qwen. Reviewers see full frontier-quality comparison without your Anthropic key on HF.

## 1. Create a Streamlit Space

1. [huggingface.co/new-space](https://huggingface.co/new-space)
2. SDK: **Streamlit**
3. Visibility: **Public**

## 2. Push your repository

Connect GitHub or clone the Space repo and push your code.

Add this YAML at the **top** of the Space `README.md`:

```yaml
---
title: AI Assistant Comparison Lab
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.36.0"
app_file: main.py
pinned: false
---
```

## 3. Space secrets (Settings → Secrets)

| Secret | Required? | Purpose |
|--------|-----------|---------|
| `HF_TOKEN` | **Yes** | OSS assistant (Qwen via HF Hub) |

**Do not add** `FRONTIER_API_KEY` or `JUDGE_API_KEY`.

Optional:

| Secret | Purpose |
|--------|---------|
| `OSS_MODEL_NAME` | Default `Qwen/Qwen2.5-0.5B-Instruct` |
| `FRONTIER_MODEL_NAME` | Shown in UI; default `claude-sonnet-4-6` when user supplies Anthropic key |

Your **Anthropic key stays in local `.env`** for development and eval runs only.

## 4. What visitors can do

- **OSS Assistant** — works immediately with Space `HF_TOKEN`
- **Frontier Assistant** — paste **their own** Anthropic key in the sidebar (session-only, never saved to disk)
- **Evaluation tab** — bundled CSV/charts/PDF from the repo (Claude vs Qwen local eval)

## 5. Local full setup (your machine)

```bash
# .env
HF_TOKEN=...
FRONTIER_API_KEY=sk-ant-...   # Claude — never commit
JUDGE_API_KEY=sk-ant-...      # optional live re-eval
FRONTIER_PROVIDER=anthropic
FRONTIER_MODEL_NAME=claude-sonnet-4-6

python scripts/run_evaluation.py
python scripts/generate_evaluation_pdf.py
```

## 6. Demo URL

`https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`
