"""Provider-agnostic OSS assistant implementation."""

from __future__ import annotations

from app.assistants.base_assistant import BaseAssistant
from app.memory import MemoryManager
from app.prompts import compose_system_prompt
from app.providers import ProviderFactory, ProviderSettings


class OSSAssistant(BaseAssistant):
    def __init__(
        self,
        provider: str,
        api_key: str | None,
        memory: MemoryManager,
        system_context: str = "",
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        temperature: float = 0.4,
        max_new_tokens: int = 512,
        base_url: str | None = None,
    ) -> None:
        super().__init__(assistant_name="OSS Assistant", model_name=model_name, memory=memory)
        self.provider = ProviderFactory.create(
            ProviderSettings(provider=provider, api_key=api_key, model_name=model_name, base_url=base_url)
        )
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system_context = system_context

    def _generate_response(self, user_message: str) -> tuple[str, dict[str, object]]:
        messages = [{"role": "system", "content": compose_system_prompt(self.system_context)}] + self.memory.to_openai_style()
        try:
            content, metadata = self.provider.generate(messages, max_tokens=self.max_new_tokens, temperature=self.temperature)
            return content, metadata
        except Exception as exc:
            fallback = "I ran into an API error while generating a response. Please try again."
            return fallback, {"provider": self.provider.__class__.__name__, "error": str(exc)}
