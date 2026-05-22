"""Streamlit user interface for live comparison and evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.assistants.frontier_assistant import FrontierAssistant
from app.assistants.oss_assistant import OSSAssistant
from app.config import load_config
from app.memory import MemoryManager
from app.runtime_config import (
    missing_keys_for_chat,
    missing_keys_for_eval,
    resolve_runtime_config,
)
from app.storage import ChatSessionStore, GlobalMemoryStore
from app.utils import ObservabilityLogger, ensure_dir, utc_now_iso
from evals.evaluator import EvaluationRunner
from evals.judge import HostedJudge


ROOT_DIR = Path(__file__).resolve().parents[1]


def initialize_state() -> None:
    defaults = {
        "active_model": "OSS Assistant",
        "active_chat_id": None,
        "memory_window": 8,
        "temperature": 0.4,
        "queued_prompt": None,
        "evaluation_results": None,
        "evaluation_summary": None,
        "evaluation_category": None,
        "evaluation_charts": {},
        "last_response_meta": {},
        "frontier_api_key_input": "",
        "judge_api_key_input": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _session_overrides() -> dict[str, str]:
    overrides: dict[str, str] = {}
    frontier_key = (st.session_state.get("frontier_api_key_input") or "").strip()
    if frontier_key:
        overrides["frontier_api_key"] = frontier_key
    judge_key = (st.session_state.get("judge_api_key_input") or "").strip()
    if judge_key:
        overrides["judge_api_key"] = judge_key
    return overrides


def build_assistant(
    name: str,
    runtime,
    memory_window: int,
    thread_messages: list[dict[str, str]],
    system_context: str = "",
):
    memory = MemoryManager(window_size=memory_window)
    memory.extend(thread_messages)
    # Prepend role marker so `compose_system_prompt` can apply role-specific instructions
    if system_context is None:
        system_context = ""
    system_context_with_role = f"[ASSISTANT_ROLE:{name}]\n" + system_context.strip()
    if name == "OSS Assistant":
        return OSSAssistant(
            provider=runtime.oss_provider,
            api_key=runtime.oss_api_key,
            memory=memory,
            system_context=system_context_with_role,
            model_name=runtime.oss_model_name,
            temperature=runtime.temperature,
            max_new_tokens=runtime.max_new_tokens,
            base_url=runtime.oss_base_url,
        )
    return FrontierAssistant(
        provider=runtime.frontier_provider,
        api_key=runtime.frontier_api_key,
        memory=memory,
        system_context=system_context_with_role,
        model_name=runtime.frontier_model_name,
        temperature=runtime.temperature,
        max_new_tokens=runtime.max_new_tokens,
        base_url=runtime.frontier_base_url,
    )


def message_dicts(session) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in session.messages]


def process_message(assistant_name: str, runtime, prompt: str, logger: ObservabilityLogger, session_store: ChatSessionStore, global_memory_store: GlobalMemoryStore):
    if assistant_name == "Frontier Assistant" and not runtime.frontier_configured:
        return None

    active_session = session_store.get_session(st.session_state.active_chat_id)
    if active_session is None:
        active_session = session_store.ensure_session(assistant_name)
        st.session_state.active_chat_id = active_session.id

    assistant = build_assistant(
        assistant_name,
        runtime,
        st.session_state.memory_window,
        thread_messages=message_dicts(active_session),
        system_context=global_memory_store.as_context_text(),
    )
    logger.log_request({"assistant": assistant_name, "prompt": prompt, "timestamp": utc_now_iso()})
    result = assistant.chat(prompt)
    session_store.append_message(active_session.id, "user", prompt)
    session_store.append_message(active_session.id, "assistant", result.content)
    if active_session.title == "New chat" and prompt.strip():
        session_store.rename_session(active_session.id, prompt.strip()[:48])
    logger.log_response(
        {
            "assistant": assistant_name,
            "response": result.content,
            "latency_ms": result.latency_ms,
            "refusal": result.refusal,
            "safety_triggered": result.safety_triggered,
        }
    )
    logger.log_latency({"assistant": assistant_name, "latency_ms": result.latency_ms})
    st.session_state.last_response_meta = {
        "assistant": assistant_name,
        "model": result.model_name,
        "latency_ms": result.latency_ms,
        "refusal": result.refusal,
        "safety_triggered": result.safety_triggered,
        "metadata": result.metadata,
    }
    return result


def render_chat_history(messages) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _load_bundled_evaluation(runtime) -> None:
    summary_path = runtime.reports_dir / "metrics_summary.csv"
    category_path = runtime.reports_dir / "metrics_by_category.csv"
    if summary_path.exists() and st.session_state.evaluation_summary is None:
        st.session_state.evaluation_summary = pd.read_csv(summary_path)
    if category_path.exists() and st.session_state.get("evaluation_category") is None:
        st.session_state.evaluation_category = pd.read_csv(category_path)
    if not st.session_state.evaluation_charts:
        charts: dict[str, Path] = {}
        for png in sorted(runtime.graphs_dir.glob("*.png")):
            charts[png.stem] = png
        st.session_state.evaluation_charts = charts


def render_evaluation_dashboard(runtime, logger: ObservabilityLogger) -> None:
    st.subheader("Evaluation Dashboard")
    _load_bundled_evaluation(runtime)

    st.write(
        "Pre-computed benchmark results (Claude vs Qwen, 76 prompts) ship with this repo. "
        "Re-run live only if you provide a judge API key below or in `.env`."
    )

    can_run_live = bool(runtime.judge_api_key)

    if can_run_live and st.button("Run live evaluation suite", type="primary"):
        judge = HostedJudge(
            provider=runtime.judge_provider,
            api_key=runtime.judge_api_key,
            model_name=runtime.judge_model_name,
            base_url=runtime.judge_base_url,
        )
        runner = EvaluationRunner(judge, runtime.reports_dir, runtime.graphs_dir)
        assistants = {
            "OSS Assistant": lambda: build_assistant("OSS Assistant", runtime, st.session_state.memory_window, thread_messages=[], system_context=""),
            "Frontier Assistant": lambda: build_assistant("Frontier Assistant", runtime, st.session_state.memory_window, thread_messages=[], system_context=""),
        }
        with st.spinner("Running benchmark suite (core + expanded prompts)..."):
            results_frame, summary_frame, category_frame, charts = runner.run(
                assistants, runtime.benchmark_dir, include_expanded=True
            )
        st.session_state.evaluation_results = results_frame
        st.session_state.evaluation_summary = summary_frame
        st.session_state.evaluation_category = category_frame
        st.session_state.evaluation_charts = charts
        logger.log_evaluation({"event": "completed", "rows": len(results_frame)})
    else:
        st.info("Live re-run disabled without a judge API key. Showing bundled results from the local Claude evaluation.")

    pdf_path = runtime.reports_dir / "evaluation_report.pdf"
    if pdf_path.exists():
        st.download_button(
            "Download evaluation report (PDF)",
            data=pdf_path.read_bytes(),
            file_name="evaluation_report.pdf",
            mime="application/pdf",
        )

    if isinstance(st.session_state.evaluation_summary, pd.DataFrame):
        st.dataframe(st.session_state.evaluation_summary, use_container_width=True)
        if isinstance(st.session_state.get("evaluation_category"), pd.DataFrame):
            with st.expander("Per-category breakdown"):
                st.dataframe(st.session_state.evaluation_category, use_container_width=True)
        st.download_button(
            "Download metrics_summary.csv",
            data=st.session_state.evaluation_summary.to_csv(index=False),
            file_name="metrics_summary.csv",
            mime="text/csv",
        )
    if isinstance(st.session_state.evaluation_results, pd.DataFrame):
        st.download_button(
            "Download benchmark_results.json",
            data=st.session_state.evaluation_results.to_json(orient="records", indent=2),
            file_name="benchmark_results.json",
            mime="application/json",
        )
    for label, path in st.session_state.evaluation_charts.items():
        if Path(path).exists():
            st.image(str(path), caption=label.replace("_", " ").title(), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="AI Assistant Comparison", page_icon="🤖", layout="wide")
    initialize_state()
    base_config = load_config()
    runtime = resolve_runtime_config(base_config, _session_overrides())
    st.session_state.setdefault("memory_window", int(runtime.memory_window))
    st.session_state.setdefault("temperature", float(runtime.temperature))
    chat_store = ChatSessionStore(runtime.data_dir / "chat_sessions.json")
    global_memory_store = GlobalMemoryStore(runtime.data_dir / "global_memory.json")
    missing = missing_keys_for_chat(runtime)
    ensure_dir(runtime.logs_dir)
    ensure_dir(runtime.data_dir)
    ensure_dir(runtime.reports_dir)
    ensure_dir(runtime.graphs_dir)
    logger = ObservabilityLogger(runtime.logs_dir)

    if missing:
        st.error("Missing for chat: " + ", ".join(missing))
        st.info("On Hugging Face Spaces, add `HF_TOKEN` under Settings → Secrets. See DEPLOY_HF.md.")
        st.stop()

    st.title("AI Assistant Comparison Lab")
    if runtime.public_demo_mode:
        st.caption(
            f"OSS: `{runtime.oss_model_name}` (HF). "
            f"Frontier: `{runtime.frontier_model_name}` (Anthropic) — paste your API key in the sidebar on this Space, "
            "or run locally with `.env`. Bundled eval compares Claude vs Qwen."
        )
    else:
        st.caption(
            f"Compare OSS `{runtime.oss_model_name}` with frontier `{runtime.frontier_model_name}` across quality and safety."
        )

    sessions = chat_store.list_sessions()
    if not sessions:
        default_session = chat_store.create_session(st.session_state.active_model)
        sessions = [default_session]
    if st.session_state.active_chat_id is None or not chat_store.get_session(st.session_state.active_chat_id):
        st.session_state.active_chat_id = sessions[0].id

    active_session = chat_store.get_session(st.session_state.active_chat_id) or sessions[0]
    st.session_state.active_model = active_session.assistant_name

    with st.sidebar:
        st.header("Configuration")
        chat_labels = {session.id: f"{session.title} · {session.assistant_name}" for session in sessions}
        selected_chat_id = st.selectbox(
            "Chats",
            options=[session.id for session in sessions],
            format_func=lambda session_id: chat_labels.get(session_id, session_id),
            index=[session.id for session in sessions].index(st.session_state.active_chat_id),
        )
        if selected_chat_id != st.session_state.active_chat_id:
            st.session_state.active_chat_id = selected_chat_id
            st.rerun()

        active_session = chat_store.get_session(st.session_state.active_chat_id) or sessions[0]
        st.session_state.active_model = st.selectbox(
            "Assistant model",
            ["OSS Assistant", "Frontier Assistant"],
            index=0 if active_session.assistant_name == "OSS Assistant" else 1,
        )
        if st.session_state.active_model != active_session.assistant_name:
            active_session.assistant_name = st.session_state.active_model
            chat_store.upsert_session(active_session)

        left, right = st.columns(2)
        with left:
            if st.button("New chat"):
                new_session = chat_store.create_session(st.session_state.active_model)
                st.session_state.active_chat_id = new_session.id
                st.rerun()
        with right:
            if st.button("Delete chat"):
                chat_store.delete_session(active_session.id)
                remaining = chat_store.list_sessions()
                if remaining:
                    st.session_state.active_chat_id = remaining[0].id
                    st.session_state.active_model = remaining[0].assistant_name
                else:
                    created = chat_store.create_session(st.session_state.active_model)
                    st.session_state.active_chat_id = created.id
                st.rerun()

        # Memory window and temperature are configured via environment (.env) or defaults in AppConfig.
        # Removed interactive sliders to keep these backend-controlled.
        if st.button("Clear chat"):
            active_session.messages = []
            chat_store.upsert_session(active_session)
            st.session_state.last_response_meta = {}
            st.rerun()

        st.divider()
        st.subheader("Global memory")
        st.caption("Facts and preferences injected into every assistant turn.")
        memory_items = global_memory_store.list_items()
        for item in memory_items:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(item.content[:120] + ("…" if len(item.content) > 120 else ""))
            with col2:
                if st.button("Delete", key=f"del_mem_{item.id}"):
                    global_memory_store.delete_item(item.id)
                    st.rerun()
        new_memory = st.text_area("Add memory", placeholder="e.g. Prefer concise answers under 3 sentences.")
        if st.button("Save memory") and new_memory.strip():
            global_memory_store.add_item(new_memory)
            st.rerun()

        st.divider()
        st.subheader("API keys (optional)")
        st.caption("Keys stay in this browser session only — never written to disk or git.")
        st.text_input(
            "Anthropic API key (Claude frontier)",
            type="password",
            key="frontier_api_key_input",
            placeholder="sk-ant-… required for Frontier Assistant",
        )
        st.text_input(
            "Judge API key (live re-eval)",
            type="password",
            key="judge_api_key_input",
            placeholder="Optional — bundled report works without this",
        )
        if not runtime.frontier_configured:
            st.warning("Frontier Assistant needs an Anthropic key above (or FRONTIER_API_KEY in `.env` locally).")

    runtime = resolve_runtime_config(base_config, _session_overrides())

    tabs = st.tabs(["Chat", "Evaluation Dashboard"])

    with tabs[0]:
        active_session = chat_store.get_session(st.session_state.active_chat_id) or sessions[0]
        render_chat_history(message_dicts(active_session))
        if st.session_state.queued_prompt:
            prompt = st.session_state.queued_prompt
            st.session_state.queued_prompt = None
            with st.chat_message("user"):
                st.markdown(prompt)
            result = process_message(st.session_state.active_model, runtime, prompt, logger, chat_store, global_memory_store)
            if result is None:
                st.error("Frontier Assistant requires an Anthropic API key (see sidebar).")
            else:
                with st.chat_message("assistant"):
                    st.markdown(result.content)
                st.success(f"{result.assistant_name} replied in {result.latency_ms} ms")

        prompt = st.chat_input("Ask either assistant anything...")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            result = process_message(st.session_state.active_model, runtime, prompt, logger, chat_store, global_memory_store)
            if result is None:
                st.error(
                    "Frontier Assistant requires an Anthropic API key. "
                    "Add it in the sidebar (session only) or set FRONTIER_API_KEY in `.env` for local runs."
                )
            else:
                with st.chat_message("assistant"):
                    st.markdown(result.content)
                st.metric("Latency (ms)", f"{result.latency_ms:.2f}")
                st.caption(f"Model: {result.model_name}")
                if result.refusal:
                    st.warning("Refusal behavior triggered.")
                if result.safety_triggered:
                    st.error("Safety guardrail blocked the request before model execution.")

    with tabs[1]:
        render_evaluation_dashboard(runtime, logger)


if __name__ == "__main__":
    main()
