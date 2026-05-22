"""Persistence helpers for chat sessions and global memory."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.utils import ensure_dir


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_load(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return default


def _atomic_json_write(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temp_path.replace(path)


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass
class ChatSession:
    id: str
    title: str
    assistant_name: str
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    messages: list[ChatMessage] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "assistant_name": self.assistant_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [asdict(message) for message in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatSession":
        return cls(
            id=str(data.get("id", uuid.uuid4())),
            title=str(data.get("title", "New chat")),
            assistant_name=str(data.get("assistant_name", "OSS Assistant")),
            created_at=str(data.get("created_at", utc_now_iso())),
            updated_at=str(data.get("updated_at", utc_now_iso())),
            messages=[ChatMessage(**message) for message in data.get("messages", []) if isinstance(message, dict)],
        )


@dataclass
class GlobalMemoryItem:
    id: str
    content: str
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalMemoryItem":
        return cls(
            id=str(data.get("id", uuid.uuid4())),
            content=str(data.get("content", "")),
            created_at=str(data.get("created_at", utc_now_iso())),
            updated_at=str(data.get("updated_at", utc_now_iso())),
        )


class ChatSessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_dir(self.path.parent)

    def _load(self) -> list[ChatSession]:
        payload = _safe_json_load(self.path, {"sessions": []})
        return [ChatSession.from_dict(item) for item in payload.get("sessions", []) if isinstance(item, dict)]

    def _save(self, sessions: list[ChatSession]) -> None:
        _atomic_json_write(self.path, {"sessions": [session.to_dict() for session in sessions]})

    def list_sessions(self) -> list[ChatSession]:
        sessions = self._load()
        return sorted(sessions, key=lambda session: session.updated_at, reverse=True)

    def get_session(self, session_id: str) -> ChatSession | None:
        for session in self._load():
            if session.id == session_id:
                return session
        return None

    def create_session(self, assistant_name: str, title: str = "New chat") -> ChatSession:
        sessions = self._load()
        session = ChatSession(id=str(uuid.uuid4()), title=title, assistant_name=assistant_name)
        sessions.append(session)
        self._save(sessions)
        return session

    def upsert_session(self, updated_session: ChatSession) -> None:
        sessions = self._load()
        updated_session.updated_at = utc_now_iso()
        replaced = False
        for index, session in enumerate(sessions):
            if session.id == updated_session.id:
                sessions[index] = updated_session
                replaced = True
                break
        if not replaced:
            sessions.append(updated_session)
        self._save(sessions)

    def delete_session(self, session_id: str) -> None:
        sessions = [session for session in self._load() if session.id != session_id]
        self._save(sessions)

    def append_message(self, session_id: str, role: str, content: str) -> ChatSession | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        session.messages.append(ChatMessage(role=role, content=content))
        session.updated_at = utc_now_iso()
        self.upsert_session(session)
        return session

    def rename_session(self, session_id: str, title: str) -> ChatSession | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        session.title = title.strip() or "New chat"
        self.upsert_session(session)
        return session

    def ensure_session(self, assistant_name: str) -> ChatSession:
        sessions = self.list_sessions()
        if sessions:
            return sessions[0]
        return self.create_session(assistant_name=assistant_name)


class GlobalMemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_dir(self.path.parent)

    def _load(self) -> list[GlobalMemoryItem]:
        payload = _safe_json_load(self.path, {"items": []})
        return [GlobalMemoryItem.from_dict(item) for item in payload.get("items", []) if isinstance(item, dict)]

    def _save(self, items: list[GlobalMemoryItem]) -> None:
        _atomic_json_write(self.path, {"items": [item.to_dict() for item in items]})

    def list_items(self) -> list[GlobalMemoryItem]:
        return sorted(self._load(), key=lambda item: item.updated_at, reverse=True)

    def add_item(self, content: str) -> GlobalMemoryItem | None:
        text = content.strip()
        if not text:
            return None
        items = self._load()
        item = GlobalMemoryItem(id=str(uuid.uuid4()), content=text)
        items.append(item)
        self._save(items)
        return item

    def update_item(self, item_id: str, content: str) -> GlobalMemoryItem | None:
        items = self._load()
        for index, item in enumerate(items):
            if item.id == item_id:
                items[index] = GlobalMemoryItem(
                    id=item.id,
                    content=content.strip(),
                    created_at=item.created_at,
                    updated_at=utc_now_iso(),
                )
                self._save(items)
                return items[index]
        return None

    def delete_item(self, item_id: str) -> None:
        items = [item for item in self._load() if item.id != item_id]
        self._save(items)

    def clear(self) -> None:
        self._save([])

    def as_context_text(self) -> str:
        items = self.list_items()
        if not items:
            return ""
        return "\n".join(f"- {item.content}" for item in items)
