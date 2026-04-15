"""LLM Gateways Python SDK.

Protect OpenAI and Anthropic API calls from prompt injection, jailbreaks,
and data-extraction attacks — with a single line of code.

Quick start::

    from llmgateways import wrap, PromptBlockedError
    from openai import OpenAI

    client = wrap(OpenAI(), api_key="lgk_...")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
        )
    except PromptBlockedError as e:
        print("Blocked:", e.result.threats)
"""
from __future__ import annotations

from typing import Any

from .client import LLMGatewaysClient
from .exceptions import LLMGatewaysError, PromptBlockedError
from .models import ScanResult

__all__ = [
    "wrap",
    "LLMGatewaysClient",
    "PromptBlockedError",
    "LLMGatewaysError",
    "ScanResult",
]

__version__ = "0.1.0"


def wrap(client: Any, *, api_key: str, base_url: str = "", timeout: float = 10.0) -> Any:
    """Wrap an OpenAI or Anthropic client with LLM Gateways protection.

    Every call to ``chat.completions.create`` (OpenAI) or ``messages.create``
    (Anthropic) will be scanned before reaching the model. Blocked prompts
    raise :exc:`PromptBlockedError`.

    Parameters
    ----------
    client:
        An ``openai.OpenAI``, ``openai.AsyncOpenAI``, ``anthropic.Anthropic``,
        or ``anthropic.AsyncAnthropic`` instance.
    api_key:
        Your ``lgk_...`` API key from https://llmgateways.com/dashboard.
    base_url:
        Override the gateway URL (for self-hosted deployments).
    timeout:
        Per-request timeout in seconds passed to the gateway (default 10 s).

    Returns
    -------
    A protected proxy with the same interface as the original client.

    Raises
    ------
    TypeError
        If the client type is not recognised.
    """
    kwargs: dict = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    gw = LLMGatewaysClient(**kwargs)

    client_type = type(client).__name__
    module = type(client).__module__

    if "openai" in module:
        from .wrappers.openai import wrap_openai
        return wrap_openai(client, gw)

    if "anthropic" in module:
        from .wrappers.anthropic import wrap_anthropic
        return wrap_anthropic(client, gw)

    raise TypeError(
        f"Unsupported client type: {client_type}. "
        "Pass an openai.OpenAI (or any OpenAI-compatible client) "
        "or an anthropic.Anthropic instance."
    )
