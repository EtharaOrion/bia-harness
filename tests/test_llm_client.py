from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pytest

from llm_client import (
    LLMClient,
    LLMError,
    LLMRateLimitError,
    _parse_response,
    _strip_provider_prefix,
    load_and_build,
)


def _sdk_message(body):
    """Stand-in for an SDK Message: only `.model_dump()` is consumed."""
    m = MagicMock()
    m.model_dump.return_value = body
    return m


def _httpx_response(status_code, json_body=None, headers=None, text=""):
    req = httpx.Request("POST", "http://127.0.0.1:9/v1/messages")
    if json_body is not None:
        return httpx.Response(status_code, request=req, json=json_body,
                              headers=headers or {})
    return httpx.Response(status_code, request=req, text=text,
                          headers=headers or {})


def _status_error(status_code, json_body=None, headers=None, text=""):
    resp = _httpx_response(status_code, json_body, headers, text)
    return anthropic.APIStatusError(f"HTTP {status_code}", response=resp,
                                    body=json_body)


def _rate_limit_error(json_body=None, headers=None):
    resp = _httpx_response(429, json_body if json_body is not None else {}, headers)
    return anthropic.RateLimitError("rate limited", response=resp, body=json_body)


def test_client_requires_fields():
    with pytest.raises(ValueError):
        LLMClient(base_url="", api_key="k", model="m")
    with pytest.raises(ValueError):
        LLMClient(base_url="u", api_key="", model="m")
    with pytest.raises(ValueError):
        LLMClient(base_url="u", api_key="k", model="")


def test_strip_provider_prefix():
    assert _strip_provider_prefix("anthropic/claude-opus-4-8") == "claude-opus-4-8"
    assert _strip_provider_prefix("claude-opus-4-8") == "claude-opus-4-8"


def test_sdk_constructed_with_owned_retry_loop():
    """max_retries=0 -- this class owns retries, the SDK must not double them."""
    with patch("llm_client.Anthropic") as sdk:
        LLMClient(base_url="http://172.17.0.1:8765/", api_key="stub-key",
                  model="anthropic/claude-opus-4-8", timeout_sec=600.0)
    kwargs = sdk.call_args.kwargs
    assert kwargs["base_url"] == "http://172.17.0.1:8765"
    assert kwargs["api_key"] == "stub-key"
    assert kwargs["timeout"] == 600.0
    assert kwargs["max_retries"] == 0


def test_messages_ok_parses_text_and_tool_use():
    body = {
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "toolu_1", "name": "write_variant", "input": {"path": "x.py"}},
        ],
        "stop_reason": "tool_use",
    }
    with patch("llm_client.Anthropic") as sdk:
        sdk.return_value.messages.create.return_value = _sdk_message(body)
        client = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m")
        resp = client.messages(system="s", messages=[{"role": "user", "content": "hi"}])
    assert resp.text_blocks == ["hello"]
    assert resp.tool_uses == [{"id": "toolu_1", "name": "write_variant", "input": {"path": "x.py"}}]
    assert resp.stop_reason == "tool_use"


def test_messages_sends_expected_kwargs():
    body = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
    tools = [{"name": "write_variant", "description": "d", "input_schema": {"type": "object"}}]
    with patch("llm_client.Anthropic") as sdk:
        create = sdk.return_value.messages.create
        create.return_value = _sdk_message(body)
        client = LLMClient(base_url="http://172.17.0.1:8765/", api_key="stub-key",
                           model="anthropic/claude-opus-4-8")
        client.messages(system="SYS", messages=[{"role": "user", "content": "hi"}],
                        tools=tools, max_tokens=1024, temperature=0.2)
    sent = create.call_args.kwargs
    assert sent["model"] == "claude-opus-4-8"
    assert sent["system"] == "SYS"
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["max_tokens"] == 1024
    assert sent["temperature"] == 0.2
    assert sent["tools"] == tools


def test_messages_omits_temperature_by_default():
    body = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
    with patch("llm_client.Anthropic") as sdk:
        create = sdk.return_value.messages.create
        create.return_value = _sdk_message(body)
        client = LLMClient(base_url="http://172.17.0.1:8765", api_key="k", model="claude-opus-5")
        client.messages(system="SYS", messages=[{"role": "user", "content": "hi"}])
    assert "temperature" not in create.call_args.kwargs
    assert "tools" not in create.call_args.kwargs


def test_server_retry_delay_prefers_header_then_bridge_body():
    from llm_client import _server_retry_delay

    assert _server_retry_delay(_httpx_response(429, {}, {"retry-after": "12"})) == 12.0
    assert _server_retry_delay(
        _httpx_response(429, {"aurora_bridge": {"retry_after_seconds": 7}})) == 7.0
    assert _server_retry_delay(_httpx_response(429, {"error": {"type": "rate_limit"}})) is None
    assert _server_retry_delay(None) is None


def test_messages_waits_for_server_requested_delay():
    ok = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
    with patch("llm_client.Anthropic") as sdk, patch("llm_client.time.sleep") as slept:
        sdk.return_value.messages.create.side_effect = [
            _rate_limit_error({"aurora_bridge": {"retry_after_seconds": 9}}),
            _sdk_message(ok),
        ]
        client = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m", num_retries=1)
        client.messages(system="s", messages=[{"role": "user", "content": "hi"}])
    assert slept.call_args[0][0] >= 9.0


def test_messages_rate_limit_floor_applies_without_server_hint():
    """No Retry-After -> floor at MIN_RATE_LIMIT_SLEEP_SEC, not sub-second backoff."""
    ok = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
    with patch("llm_client.Anthropic") as sdk, patch("llm_client.time.sleep") as slept:
        sdk.return_value.messages.create.side_effect = [
            _rate_limit_error({}),
            _sdk_message(ok),
        ]
        client = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m", num_retries=1)
        client.messages(system="s", messages=[{"role": "user", "content": "hi"}])
    assert slept.call_args[0][0] >= 30.0


def test_messages_rate_limit_retries_then_raises():
    with patch("llm_client.Anthropic") as sdk, patch("llm_client.time.sleep"):
        create = sdk.return_value.messages.create
        create.side_effect = _rate_limit_error({})
        client = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m", num_retries=2)
        with pytest.raises(LLMRateLimitError):
            client.messages(system="s", messages=[{"role": "user", "content": "hi"}])
        assert create.call_count == 3


def test_messages_rate_limit_then_success():
    body = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
    with patch("llm_client.Anthropic") as sdk, patch("llm_client.time.sleep"):
        sdk.return_value.messages.create.side_effect = [
            _rate_limit_error({}),
            _sdk_message(body),
        ]
        client = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m", num_retries=2)
        resp = client.messages(system="s", messages=[{"role": "user", "content": "hi"}])
    assert resp.text_blocks == ["ok"]


def test_messages_529_is_retryable():
    body = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
    with patch("llm_client.Anthropic") as sdk, patch("llm_client.time.sleep"):
        sdk.return_value.messages.create.side_effect = [
            _status_error(529, {}, text="overloaded"),
            _sdk_message(body),
        ]
        client = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m", num_retries=2)
        resp = client.messages(system="s", messages=[{"role": "user", "content": "hi"}])
    assert resp.text_blocks == ["ok"]


def test_messages_4xx_raises_immediately():
    with patch("llm_client.Anthropic") as sdk:
        create = sdk.return_value.messages.create
        create.side_effect = _status_error(400, {"error": {"message": "bad request"}})
        client = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m")
        with pytest.raises(LLMError) as exc:
            client.messages(system="s", messages=[{"role": "user", "content": "hi"}])
        assert create.call_count == 1
    assert not isinstance(exc.value, LLMRateLimitError)
    assert "400" in str(exc.value)


def test_messages_timeout_retries_then_raises():
    req = httpx.Request("POST", "http://127.0.0.1:9/v1/messages")
    with patch("llm_client.Anthropic") as sdk, patch("llm_client.time.sleep"):
        create = sdk.return_value.messages.create
        create.side_effect = anthropic.APITimeoutError(request=req)
        client = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m", num_retries=1)
        with pytest.raises(LLMError) as exc:
            client.messages(system="s", messages=[{"role": "user", "content": "hi"}])
        assert create.call_count == 2
    assert not isinstance(exc.value, LLMRateLimitError)


def test_messages_connection_error_then_success():
    body = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
    req = httpx.Request("POST", "http://127.0.0.1:9/v1/messages")
    with patch("llm_client.Anthropic") as sdk, patch("llm_client.time.sleep"):
        sdk.return_value.messages.create.side_effect = [
            anthropic.APIConnectionError(request=req),
            _sdk_message(body),
        ]
        client = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m", num_retries=2)
        resp = client.messages(system="s", messages=[{"role": "user", "content": "hi"}])
    assert resp.text_blocks == ["ok"]


def test_load_and_build_ok(tmp_path):
    cfg = tmp_path / "cc.json"
    cfg.write_text(json.dumps({
        "model": "anthropic/claude-opus-4-8",
        "base_url": "http://172.17.0.1:8765",
        "api_key": "k",
        "timeout": 600,
        "num_retries": 2,
    }))
    client = load_and_build(cfg)
    assert client.model == "claude-opus-4-8"
    assert client.base_url == "http://172.17.0.1:8765"
    assert client.timeout_sec == 600.0
    assert client.num_retries == 2


def test_load_and_build_missing_field(tmp_path):
    cfg = tmp_path / "cc.json"
    cfg.write_text(json.dumps({"model": "m", "base_url": "u"}))
    with pytest.raises(ValueError, match="api_key"):
        load_and_build(cfg)


def test_parse_response_empty():
    resp = _parse_response({"content": [], "stop_reason": "end_turn"})
    assert resp.text_blocks == []
    assert resp.tool_uses == []
