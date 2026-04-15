from __future__ import annotations

from typing import Optional

import httpx

from .exceptions import LLMGatewaysError
from .models import ScanResult

_DEFAULT_BASE_URL = "https://llmgateways-backend-nqpl7yvf3a-ew.a.run.app"
_SCAN_PATH = "/api/v1/prompt/scan"
_DEFAULT_TIMEOUT = 10.0


class LLMGatewaysClient:
    """Low-level HTTP client for the LLM Gateways scan API.

    Parameters
    ----------
    api_key:
        Your ``lgk_...`` API key from the LLM Gateways dashboard.
    base_url:
        Override the default gateway URL (useful for self-hosted deployments).
    timeout:
        Request timeout in seconds (default 10 s).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._headers = {"X-API-Key": self._api_key, "Content-Type": "application/json"}

    # ── Sync ─────────────────────────────────────────────────────────────────

    def scan(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> ScanResult:
        """Scan a prompt synchronously.

        Parameters
        ----------
        prompt:
            The user's message to scan.
        system_prompt:
            Optional system prompt to include in the scan context.
        model:
            Optional model name (informational only, does not affect detection).

        Returns
        -------
        ScanResult
            The detection result.

        Raises
        ------
        LLMGatewaysError
            On HTTP errors or network failures.
        """
        payload: dict = {"prompt": prompt}
        if system_prompt is not None:
            payload["system_prompt"] = system_prompt
        if model is not None:
            payload["model"] = model

        try:
            with httpx.Client(timeout=self._timeout) as http:
                response = http.post(
                    f"{self._base_url}{_SCAN_PATH}",
                    json=payload,
                    headers=self._headers,
                )
            response.raise_for_status()
            return ScanResult._from_dict(response.json())
        except httpx.HTTPStatusError as exc:
            raise LLMGatewaysError(
                f"Gateway returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMGatewaysError(f"Network error contacting LLM Gateways: {exc}") from exc

    # ── Async ─────────────────────────────────────────────────────────────────

    async def scan_async(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> ScanResult:
        """Async version of :meth:`scan`."""
        payload: dict = {"prompt": prompt}
        if system_prompt is not None:
            payload["system_prompt"] = system_prompt
        if model is not None:
            payload["model"] = model

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as http:
                response = await http.post(
                    f"{self._base_url}{_SCAN_PATH}",
                    json=payload,
                    headers=self._headers,
                )
            response.raise_for_status()
            return ScanResult._from_dict(response.json())
        except httpx.HTTPStatusError as exc:
            raise LLMGatewaysError(
                f"Gateway returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMGatewaysError(f"Network error contacting LLM Gateways: {exc}") from exc
