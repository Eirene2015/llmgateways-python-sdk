"""Anthropic client wrapper — intercepts messages.create before the call reaches Anthropic."""
from __future__ import annotations

from typing import Any, Optional

from ..client import LLMGatewaysClient
from ..exceptions import PromptBlockedError


def _extract_prompt(messages: list, system: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Return (last_user_message, system_prompt) from an Anthropic messages array."""
    last_user = ""
    for msg in messages:
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role == "user":
            if isinstance(content, list):
                last_user = " ".join(
                    p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")
                    for p in content
                    if (p.get("type") if isinstance(p, dict) else getattr(p, "type", "")) == "text"
                )
            else:
                last_user = str(content) if content else ""
    return last_user, system


class _ProtectedMessages:
    def __init__(self, messages: Any, gw: LLMGatewaysClient) -> None:
        self._messages = messages
        self._gw = gw

    def create(self, *, messages: list, model: str = "", system: Optional[str] = None, **kwargs: Any) -> Any:
        prompt, system_prompt = _extract_prompt(messages, system)
        result = self._gw.scan(prompt, system_prompt=system_prompt, model=model or None)
        if result.action == "block":
            raise PromptBlockedError(result)
        return self._messages.create(messages=messages, model=model, system=system, **kwargs)

    async def create_async(self, *, messages: list, model: str = "", system: Optional[str] = None, **kwargs: Any) -> Any:
        prompt, system_prompt = _extract_prompt(messages, system)
        result = await self._gw.scan_async(prompt, system_prompt=system_prompt, model=model or None)
        if result.action == "block":
            raise PromptBlockedError(result)
        import inspect
        call = self._messages.create(messages=messages, model=model, system=system, **kwargs)
        if inspect.isawaitable(call):
            return await call
        return call


class ProtectedAnthropic:
    """A thin proxy around an Anthropic client that scans every prompt."""

    def __init__(self, anthropic_client: Any, gw: LLMGatewaysClient) -> None:
        self._client = anthropic_client
        self.messages = _ProtectedMessages(anthropic_client.messages, gw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_anthropic(anthropic_client: Any, gw: LLMGatewaysClient) -> ProtectedAnthropic:
    return ProtectedAnthropic(anthropic_client, gw)
