"""Lightweight short-term conversation memory."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class MemoryManager:
    """Maintain the most recent user/assistant messages only."""

    window_size: int = 8
    messages: list[dict[str, str]] = field(default_factory=list)

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._truncate()

    def extend(self, items: Iterable[dict[str, str]]) -> None:
        for item in items:
            self.messages.append({"role": item["role"], "content": item["content"]})
        self._truncate()

    def clear(self) -> None:
        self.messages.clear()

    def snapshot(self) -> list[dict[str, str]]:
        return list(self.messages)

    def _truncate(self) -> None:
        max_messages = max(self.window_size * 2, 2)
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def to_openai_style(self) -> list[dict[str, str]]:
        return self.snapshot()
