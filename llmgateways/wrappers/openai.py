"""OpenAI client wrapper — intercepts chat.completions.create before the call reaches OpenAI."""
from __future__ import annotations

from typing import Any, Optional

from ..client import LLMGatewaysClient
from ..exceptions import PromptBlockedError


def _extract_prompt(messages: list) -> tuple[str, Optional[str]]:
    """Return (last_user_message, system_prompt) from an OpenAI messages array."""
    system_prompt: Optional[str] = None
    last_user: str = ""
    for msg in messages:
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role == "system":
            system_prompt = str(content) if content else None
        elif role == "user":
            # Handle string or list-of-parts content
            if isinstance(content, list):
                last_user = " ".join(
                    p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")
                    for p in content
                    if (p.get("type") if isinstance(p, dict) else getattr(p, "type", "")) == "text"
                )
            else:
                last_user = str(content) if content else ""
    return last_user, system_prompt


class _ProtectedCompletions:
    def __init__(self, completions: Any, gw: LLMGatewaysClient) -> None:
        self._completions = completions
        self._gw = gw

    def create(self, *, messages: list, model: str = "", **kwargs: Any) -> Any:
        prompt, system_prompt = _extract_prompt(messages)
        result = self._gw.scan(prompt, system_prompt=system_prompt, model=model or None)
        if result.action == "block":
            raise PromptBlockedError(result)
        return self._completions.create(messages=messages, model=model, **kwargs)

    async def create_async(self, *, messages: list, model: str = "", **kwargs: Any) -> Any:
        prompt, system_prompt = _extract_prompt(messages)
        result = await self._gw.scan_async(prompt, system_prompt=system_prompt, model=model or None)
        if result.action == "block":
            raise PromptBlockedError(result)
        # Support both sync and async underlying clients
        import inspect
        call = self._completions.create(messages=messages, model=model, **kwargs)
        if inspect.isawaitable(call):
            return await call
        return call


class _ProtectedChat:
    def __init__(self, chat: Any, gw: LLMGatewaysClient) -> None:
        self.completions = _ProtectedCompletions(chat.completions, gw)


class ProtectedOpenAI:
    """A thin proxy around an OpenAI client that scans every prompt."""

    def __init__(self, openai_client: Any, gw: LLMGatewaysClient) -> None:
        self._client = openai_client
        self.chat = _ProtectedChat(openai_client.chat, gw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_openai(openai_client: Any, gw: LLMGatewaysClient) -> ProtectedOpenAI:
    return ProtectedOpenAI(openai_client, gw)
