from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ScanResult


class LLMGatewaysError(Exception):
    """Base exception for all LLM Gateways SDK errors."""


class PromptBlockedError(LLMGatewaysError):
    """Raised when the gateway blocks a prompt.

    Attributes
    ----------
    result:
        The full :class:`~llmgateways.models.ScanResult` from the gateway,
        including ``risk_score``, ``threats``, ``layer_used``, and ``reasoning``.
    """

    def __init__(self, result: "ScanResult") -> None:
        self.result = result
        threats = ", ".join(result.threats) if result.threats else "unspecified"
        super().__init__(
            f"Prompt blocked by LLM Gateways "
            f"(risk={result.risk_score:.2f}, threats=[{threats}])"
        )
