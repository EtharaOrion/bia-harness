from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_client import (
    LLMClient,
    LLMError,
    LLMRateLimitError,
    _parse_response,
    _strip_provider_prefix,
    load_and_build,
)


def _mock_response(status_code=200, json_body=None, text=""):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_body or {}
    m.text = text
    return m


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


def test_messages_ok_parses_text_and_tool_use():
    body = {
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "toolu_1", "name": "write_variant", "input": {"path": "x.py"}},
        ],
        "stop_reason": "tool_use",
    }
    client = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m")
    with patch("llm_client.httpx.Client") as mock_client_cls:
        instance = mock_client_cls.return_value.__enter__.return_value
        instance.post.return_value = _mock_response(200, body)
        resp = client.messages(system="s", messages=[{"role": "user", "content": "hi"}])
    assert resp.text_blocks == ["hello"]
    assert resp.tool_uses == [{"id": "toolu_1", "name": "write_variant", "input": {"path": "x.py"}}]
    assert resp.stop_reason == "tool_use"


def test_messages_sends_expected_body_and_headers():
    body = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
    client = LLMClient(base_url="http://172.17.0.1:8765/", api_key="stub-key", model="anthropic/claude-opus-4-8")
    tools = [{"name": "write_variant", "description": "d", "input_schema": {"type": "object"}}]
    with patch("llm_client.httpx.Client") as mock_client_cls:
        instance = mock_client_cls.return_value.__enter__.return_value
        instance.post.return_value = _mock_response(200, body)
        client.messages(system="SYS", messages=[{"role": "user", "content": "hi"}], tools=tools, max_tokens=1024, temperature=0.2)
    (url,), kwargs = instance.post.call_args
    assert url == "http://172.17.0.1:8765/v1/messages"
    assert kwargs["headers"]["x-api-key"] == "stub-key"
    assert kwargs["headers"]["anthropic-version"] == "2023-06-01"
    sent = kwargs["json"]
    assert sent["model"] == "claude-opus-4-8"
    assert sent["system"] == "SYS"
    assert sent["max_tokens"] == 1024
    assert sent["temperature"] == 0.2
    assert sent["tools"] == tools


def test_messages_rate_limit_retries_then_raises():
    client = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m", num_retries=2)
    with patch("llm_client.httpx.Client") as mock_client_cls, patch("llm_client.time.sleep"):
        instance = mock_client_cls.return_value.__enter__.return_value
        instance.post.return_value = _mock_response(429, {}, "rate limit")
        with pytest.raises(LLMRateLimitError):
            client.messages(system="s", messages=[{"role": "user", "content": "hi"}])
        assert instance.post.call_count == 3


def test_messages_rate_limit_then_success():
    body = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
    client = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m", num_retries=2)
    with patch("llm_client.httpx.Client") as mock_client_cls, patch("llm_client.time.sleep"):
        instance = mock_client_cls.return_value.__enter__.return_value
        instance.post.side_effect = [
            _mock_response(429, {}, "slow down"),
            _mock_response(200, body),
        ]
        resp = client.messages(system="s", messages=[{"role": "user", "content": "hi"}])
    assert resp.text_blocks == ["ok"]


def test_messages_4xx_raises_immediately():
    client = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m")
    with patch("llm_client.httpx.Client") as mock_client_cls:
        instance = mock_client_cls.return_value.__enter__.return_value
        instance.post.return_value = _mock_response(400, {}, "bad request")
        with pytest.raises(LLMError):
            client.messages(system="s", messages=[{"role": "user", "content": "hi"}])
        assert instance.post.call_count == 1


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
