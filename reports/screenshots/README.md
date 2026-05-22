# Demo Screenshots

Captured from the local Streamlit app (`streamlit run main.py`).

| File | Description |
|------|-------------|
| `chat_home.png` / `01_home_chat.png` | Chat tab with conversation history |
| `evaluation_dashboard.png` / `02_evaluation_dashboard.png` | Evaluation Dashboard tab |
| `chat_frontier.png` / `03_chat_frontier.png` | Frontier Assistant selected |
| `chat_oss.png` / `04_chat_oss.png` | OSS Assistant selected |
| `memory_demo.png` / `05_sidebar_memory.png` | Sidebar with global memory UI |

## Regenerate

```bash
streamlit run main.py
# In another terminal (wait until UI shows "AI Assistant Comparison Lab"):
pip install playwright
python -m playwright install chromium
python scripts/capture_demo_screenshots.py
```

The capture script waits up to ~2 minutes for Streamlit to finish first-load rendering.
