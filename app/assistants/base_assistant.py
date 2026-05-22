"""Common assistant interface and shared response model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter

from app.memory import MemoryManager
from app.utils import detect_harmful_prompt, elapsed_ms, refusal_response


@dataclass
class AssistantResponse:
    assistant_name: str
    model_name: str
    content: str
    latency_ms: float
    refusal: bool = False
    safety_triggered: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


class BaseAssistant(ABC):
    """Shared behavior for both assistants."""

    def __init__(self, assistant_name: str, model_name: str, memory: MemoryManager) -> None:
        self.assistant_name = assistant_name
        self.model_name = model_name
        self.memory = memory

    def chat(self, user_message: str) -> AssistantResponse:
        safety_triggered, reason = detect_harmful_prompt(user_message)
        start = perf_counter()
        if safety_triggered:
            content = refusal_response(reason)
            latency = elapsed_ms(start)
            self.memory.append("user", user_message)
            self.memory.append("assistant", content)
            return AssistantResponse(
                assistant_name=self.assistant_name,
                model_name=self.model_name,
                content=content,
                latency_ms=latency,
                refusal=True,
                safety_triggered=True,
                metadata={"trigger": reason, "source": "guardrail"},
            )

        self.memory.append("user", user_message)
        content, metadata = self._generate_response(user_message)
        latency = elapsed_ms(start)
        self.memory.append("assistant", content)
        refusal = self._looks_like_refusal(content)
        return AssistantResponse(
            assistant_name=self.assistant_name,
            model_name=self.model_name,
            content=content,
            latency_ms=latency,
            refusal=refusal,
            safety_triggered=False,
            metadata=metadata,
        )

    @abstractmethod
    def _generate_response(self, user_message: str) -> tuple[str, dict[str, object]]:
        raise NotImplementedError

    @staticmethod
    def _looks_like_refusal(content: str) -> bool:
        lowered = content.lower()
        return any(token in lowered for token in ["can't help", "cannot help", "i can’t help", "safe alternative"])
