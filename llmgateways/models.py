from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ScanResult:
    """Result returned by the LLM Gateways detection engine."""

    risk_score: float
    """0.0 – 1.0 probability that the prompt is an attack."""

    action: str
    """``"allow"`` or ``"block"``."""

    threats: List[str] = field(default_factory=list)
    """Detected threat categories, e.g. ``["jailbreak", "injection"]``."""

    latency_ms: int = 0
    """Time the gateway spent scanning, in milliseconds."""

    layer_used: int = 1
    """Detection layer that made the decision (1 = patterns, 2 = semantic, 3 = LLM judge)."""

    reasoning: Optional[str] = None
    """Human-readable explanation, populated when layer 3 is used."""

    @classmethod
    def _from_dict(cls, data: dict) -> "ScanResult":
        return cls(
            risk_score=float(data.get("risk_score", 0.0)),
            action=str(data.get("action", "allow")),
            threats=list(data.get("threats", [])),
            latency_ms=int(data.get("latency_ms", 0)),
            layer_used=int(data.get("layer_used", 1)),
            reasoning=data.get("reasoning"),
        )
