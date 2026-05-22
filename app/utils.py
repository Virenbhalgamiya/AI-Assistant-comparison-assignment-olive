"""Shared utilities for safety, logging, export, and formatting."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from app.prompts import REFUSAL_TEMPLATE, SAFETY_KEYWORDS, SAFETY_REGEX


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000.0, 2)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_harmful_prompt(text: str) -> tuple[bool, str | None]:
    lowered = text.lower()
    for keyword in SAFETY_KEYWORDS:
        if keyword in lowered:
            return True, f"keyword:{keyword}"
    for pattern in SAFETY_REGEX:
        if re.search(pattern, lowered, re.IGNORECASE):
            return True, f"pattern:{pattern}"
    return False, None


def refusal_response(reason: str | None = None) -> str:
    if reason:
        return f"{REFUSAL_TEMPLATE} (safety trigger: {reason})"
    return REFUSAL_TEMPLATE


def safe_json_loads(raw: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = fallback or {}
    try:
        candidate = raw.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1:
            candidate = candidate[start : end + 1]
        return json.loads(candidate)
    except Exception:
        return fallback


@dataclass
class LogRecord:
    timestamp: str
    event_type: str
    payload: dict[str, Any]


class ObservabilityLogger:
    """Append structured JSONL logs for request, response, and evaluation events."""

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = ensure_dir(logs_dir)
        self.logger = logging.getLogger("ai_assistant_comparison")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _write_jsonl(self, file_name: str, record: LogRecord) -> None:
        path = self.logs_dir / file_name
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def log_request(self, payload: dict[str, Any]) -> None:
        record = LogRecord(timestamp=utc_now_iso(), event_type="request", payload=payload)
        self._write_jsonl("requests.jsonl", record)

    def log_response(self, payload: dict[str, Any]) -> None:
        record = LogRecord(timestamp=utc_now_iso(), event_type="response", payload=payload)
        self._write_jsonl("responses.jsonl", record)

    def log_evaluation(self, payload: dict[str, Any]) -> None:
        record = LogRecord(timestamp=utc_now_iso(), event_type="evaluation", payload=payload)
        self._write_jsonl("evaluations.jsonl", record)

    def log_latency(self, payload: dict[str, Any]) -> None:
        record = LogRecord(timestamp=utc_now_iso(), event_type="latency", payload=payload)
        self._write_jsonl("latency.jsonl", record)


def export_dataframe(frame: pd.DataFrame, csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    frame.to_csv(csv_path, index=False)


def dump_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
