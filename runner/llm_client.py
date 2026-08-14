from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


MAX_RETRY_SLEEP_SEC = 60.0


class LLMError(RuntimeError):
    pass


class LLMRateLimitError(LLMError):
    pass


@dataclass
class LLMResponse:
    text_blocks: list[str]
    tool_uses: list[dict[str, Any]]
    stop_reason: str | None
    raw: dict[str, Any]


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_sec: float = 300.0,
        num_retries: int = 4,
    ) -> None:
        if not base_url:
            raise ValueError("base_url required")
        if not api_key:
            raise ValueError("api_key required")
        if not model:
            raise ValueError("model required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = _strip_provider_prefix(model)
        self.timeout_sec = timeout_sec
        self.num_retries = num_retries

    def messages(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # Newer models reject `temperature` outright, so it is only sent when asked for.
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = tools

        url = f"{self.base_url}/v1/messages"
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        delay = 1.0
        for attempt in range(self.num_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_sec) as client:
                    resp = client.post(url, headers=headers, json=body)
            except httpx.TimeoutException as e:
                if attempt >= self.num_retries:
                    raise LLMError(f"timeout after {self.num_retries + 1} attempts: {e}") from e
                time.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 2, 30.0)
                continue

            if resp.status_code in (429, 529):
                server_wait = _server_retry_delay(resp)
                if attempt >= self.num_retries:
                    hint = f"; server asked for {server_wait:.0f}s" if server_wait else ""
                    raise LLMRateLimitError(
                        f"rate-limited (HTTP {resp.status_code}) after "
                        f"{self.num_retries + 1} attempts{hint}"
                    )
                wait = server_wait if server_wait is not None else delay
                time.sleep(min(wait, MAX_RETRY_SLEEP_SEC) + random.uniform(0, 0.5))
                delay = min(delay * 2, MAX_RETRY_SLEEP_SEC)
                continue

            if resp.status_code >= 400:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")

            return _parse_response(resp.json())

        raise LLMError("unreachable retry loop")


def _server_retry_delay(resp) -> float | None:
    """Seconds the server asked us to wait, from the Retry-After header or bridge body."""
    raw = resp.headers.get("retry-after")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    try:
        payload = resp.json()
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    seconds = (payload.get("aurora_bridge") or {}).get("retry_after_seconds")
    if isinstance(seconds, (int, float)):
        return max(0.0, float(seconds))
    return None


def _strip_provider_prefix(model: str) -> str:
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def _parse_response(data: dict[str, Any]) -> LLMResponse:
    text_blocks: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    for block in data.get("content", []):
        btype = block.get("type")
        if btype == "text":
            text_blocks.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_uses.append({
                "id": block.get("id"),
                "name": block.get("name"),
                "input": block.get("input", {}),
            })
    return LLMResponse(
        text_blocks=text_blocks,
        tool_uses=tool_uses,
        stop_reason=data.get("stop_reason"),
        raw=data,
    )


def load_and_build(llm_config_path: str | Path) -> LLMClient:
    p = Path(llm_config_path)
    cfg = json.loads(p.read_text())
    for k in ("model", "base_url", "api_key"):
        if k not in cfg:
            raise ValueError(f"llm-config missing required field: {k}")
    return LLMClient(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        timeout_sec=float(cfg.get("timeout", 300.0)),
        num_retries=int(cfg.get("num_retries", 4)),
    )
