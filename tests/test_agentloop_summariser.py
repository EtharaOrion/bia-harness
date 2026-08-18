"""Tests for agentloop.summariser, ported from track3-pipeline/tools/summarize.py.

This module writes the prose that is injected into the NEXT iteration's prompt,
so the failure modes that matter are not "does it call the API" but:

  * does it INVENT structure -- extra keys, missing keys, unbounded fields;
  * does it LEAK an outage as if it were a real summary (the `_degraded` marker);
  * does it CRASH the loop when the summariser is down.

Every field cap, the fallback marker, and the degraded render path are therefore
pinned independently rather than only through the happy path.

NO TEST MAKES A REAL NETWORK CALL. The autouse `_no_network` fixture below
replaces `urllib.request.urlopen` with a blocker that raises on any un-mocked
use, so a regression that reintroduces live HTTP fails the suite structurally
rather than by hanging on a 600s timeout.
"""

import json

import pytest

from agentloop.summariser import (
    API_KEY_STUB,
    BACKOFF_BASE_SEC,
    BACKOFF_MAX_SEC,
    BRIDGE,
    FIELD_CAPS,
    INSTRUCTIONS,
    MAX_MESSAGES,
    MAX_TRAJECTORY_CHARS,
    MODEL,
    RETRIES,
    SCHEMA_KEYS,
    _clip,
    _coerce,
    ask,
    fallback,
    render,
    summarize_iteration,
    trajectory_slice,
)
import agentloop.summariser as summariser


# --- network firewall ---------------------------------------------------


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any urlopen not explicitly mocked by a test is a bug."""

    def _blocked(*a, **k):  # pragma: no cover - only runs on regression
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(summariser.urllib.request, "urlopen", _blocked)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retry backoff is real seconds; assert on the schedule instead of waiting."""
    waits = []
    monkeypatch.setattr(summariser.time, "sleep", waits.append)
    return waits


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _reply(text: str) -> bytes:
    return json.dumps({"content": [{"type": "text", "text": text}]}).encode()


VALID_JSON = json.dumps({
    "hypothesis": "Muon on the hidden matrices should cut steps-to-target.",
    "mechanism": "Replaced AdamW with Newton-Schulz orthogonalized momentum.",
    "hyperparameters": "lr=0.05, momentum=0.95, ns_steps=5",
    "measurements": "val_loss 3.28 at step 1750; baseline 3.28 at step 2100.",
    "failure_cause": "no_step_clears: no step met the target on both seeds.",
    "do_differently": "Warm up the orthogonalization over the first 100 steps.",
})


# --- constants are contract --------------------------------------------


def test_endpoint_and_model_constants():
    assert BRIDGE == "http://127.0.0.1:8765/v1/messages"
    assert MODEL == "claude-opus-5"
    assert API_KEY_STUB == "oauth-placeholder"
    assert RETRIES == 5
    assert (BACKOFF_BASE_SEC, BACKOFF_MAX_SEC) == (20, 300)
    assert MAX_TRAJECTORY_CHARS == 14000
    assert MAX_MESSAGES == 6


def test_field_caps_and_schema_keys():
    assert FIELD_CAPS == {
        "hypothesis": 400,
        "mechanism": 650,
        "hyperparameters": 500,
        "measurements": 750,
        "failure_cause": 450,
        "do_differently": 350,
    }
    assert SCHEMA_KEYS == tuple(FIELD_CAPS)
    assert len(SCHEMA_KEYS) == 6


def test_instructions_forbid_judging_and_inventing():
    """The two prohibitions that keep the summary from misleading the next attempt."""
    assert "You do NOT judge whether the attempt succeeded." in INSTRUCTIONS
    assert "Never invent numbers." in INSTRUCTIONS
    assert "Never claim the attempt was good, promising, or close." in INSTRUCTIONS


# --- _coerce ------------------------------------------------------------


def test_coerce_returns_exactly_the_six_keys_within_caps():
    out = _coerce(VALID_JSON)
    assert tuple(out) == SCHEMA_KEYS
    assert set(out) == set(SCHEMA_KEYS)
    for k, v in out.items():
        assert isinstance(v, str)
        assert len(v) <= FIELD_CAPS[k], k


def test_coerce_tolerates_prose_around_the_json():
    out = _coerce(f"Sure, here you go:\n```json\n{VALID_JSON}\n```\nHope that helps.")
    assert out["hyperparameters"] == "lr=0.05, momentum=0.95, ns_steps=5"


def test_coerce_without_json_object_raises_valueerror():
    with pytest.raises(ValueError):
        _coerce("I'm sorry, I cannot summarise that attempt.")


def test_coerce_collapses_whitespace_and_newlines():
    out = _coerce(json.dumps({"hypothesis": "  line one\n\n\tline   two  \n"}))
    assert out["hypothesis"] == "line one line two"


def test_coerce_fills_missing_keys_with_empty_string():
    out = _coerce(json.dumps({"hypothesis": "only this one"}))
    assert tuple(out) == SCHEMA_KEYS
    assert out["mechanism"] == ""
    assert out["do_differently"] == ""


def test_coerce_drops_extra_keys_the_model_invented():
    out = _coerce(json.dumps({"hypothesis": "h", "verdict": "promising!", "reward": 0.9}))
    assert "verdict" not in out
    assert "reward" not in out


def test_coerce_stringifies_non_string_values():
    out = _coerce(json.dumps({"hyperparameters": {"lr": 0.05}, "measurements": [1, 2]}))
    assert out["hyperparameters"] == '{"lr": 0.05}'
    assert out["measurements"] == "[1, 2]"


def test_coerce_clips_overlong_fields_to_cap():
    huge = {k: "word " * 500 for k in SCHEMA_KEYS}
    out = _coerce(json.dumps(huge))
    for k, v in out.items():
        # cap chars of content plus the one-character ellipsis marker
        assert len(v) <= FIELD_CAPS[k] + 1, k
        assert v.endswith("…"), k


# --- _clip --------------------------------------------------------------


def test_clip_marks_truncation_and_respects_cap():
    out = _clip("a" * 1000, 400)
    assert out.endswith("…")
    assert len(out) <= 401


def test_clip_leaves_short_text_untouched():
    assert _clip("short enough", 400) == "short enough"
    assert _clip("a" * 400, 400) == "a" * 400


def test_clip_breaks_on_a_word_boundary_when_one_is_late_enough():
    text = "word " * 100
    out = _clip(text, 50)
    assert out.endswith("…")
    assert not out.endswith(" …")


def test_clip_keeps_hard_cut_when_the_last_space_is_too_early():
    text = "hi " + "a" * 100
    out = _clip(text, 50)
    assert len(out) == 51  # hard cut at cap, no word-boundary rescue


# --- trajectory_slice ---------------------------------------------------


def _write_trajectory(trial, steps):
    agent = trial / "agent"
    agent.mkdir(parents=True, exist_ok=True)
    (agent / "trajectory.json").write_text(json.dumps({"steps": steps}))


def test_trajectory_slice_joins_long_steps_in_order(tmp_path):
    msgs = [f"step{i} " + "x" * 300 for i in range(3)]
    _write_trajectory(tmp_path, [{"message": m} for m in msgs])
    out = trajectory_slice(tmp_path)
    assert out == "\n\n---\n\n".join(msgs)
    assert out.index("step0") < out.index("step1") < out.index("step2")


def test_trajectory_slice_missing_file_returns_empty(tmp_path):
    assert trajectory_slice(tmp_path) == ""


def test_trajectory_slice_malformed_json_returns_empty(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir(parents=True)
    (agent / "trajectory.json").write_text("{not json")
    assert trajectory_slice(tmp_path) == ""


def test_trajectory_slice_skips_short_and_non_string_messages(tmp_path):
    long_msg = "keep " + "x" * 300
    _write_trajectory(tmp_path, [
        {"message": "too short"},
        {"message": {"role": "user"}},
        {"message": None},
        {"message": long_msg},
    ])
    assert trajectory_slice(tmp_path) == long_msg


def test_trajectory_slice_keeps_only_the_last_six_messages(tmp_path):
    msgs = [f"step{i} " + "x" * 300 for i in range(10)]
    _write_trajectory(tmp_path, [{"message": m} for m in msgs])
    out = trajectory_slice(tmp_path)
    assert out.count("\n\n---\n\n") == MAX_MESSAGES - 1
    assert "step9" in out and "step4" in out
    assert "step3 " not in out


def test_trajectory_slice_respects_the_total_char_budget(tmp_path):
    msgs = ["y" * 9000, "z" * 9000]
    _write_trajectory(tmp_path, [{"message": m} for m in msgs])
    out = trajectory_slice(tmp_path)
    assert len(out.replace("\n\n---\n\n", "")) == MAX_TRAJECTORY_CHARS


# --- ask ----------------------------------------------------------------


def test_ask_sends_api_key_header_and_returns_text(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["req"] = req
        seen["timeout"] = timeout
        return _FakeResponse(_reply("hello from the bridge"))

    monkeypatch.setattr(summariser.urllib.request, "urlopen", fake_urlopen)

    assert ask("summarise this") == "hello from the bridge"

    req = seen["req"]
    assert req.full_url == BRIDGE
    assert seen["timeout"] == 600
    # urllib title-cases header keys; the judge module deliberately omits x-api-key,
    # this one deliberately sends it. Do not harmonize.
    assert req.headers["X-api-key"] == "oauth-placeholder"
    assert req.headers["Content-type"] == "application/json"
    assert req.headers["Anthropic-version"] == "2023-06-01"

    body = json.loads(req.data)
    assert body["model"] == MODEL
    assert body["messages"][0]["content"].startswith(INSTRUCTIONS)
    assert "summarise this" in body["messages"][0]["content"]


def test_ask_retries_then_raises_with_documented_backoff(monkeypatch, _no_sleep):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        raise OSError("connection refused")

    monkeypatch.setattr(summariser.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="summariser unreachable after 5 attempts"):
        ask("x")

    assert len(calls) == RETRIES
    assert _no_sleep == [20, 40, 80, 160]  # min(20 * 2**attempt, 300)


def test_ask_rejects_a_reply_without_content(monkeypatch, _no_sleep):
    monkeypatch.setattr(
        summariser.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResponse(json.dumps({"error": "overloaded"}).encode()))
    with pytest.raises(RuntimeError):
        ask("x", retries=1)


# --- fallback -----------------------------------------------------------


def test_fallback_marks_itself_degraded(tmp_path):
    out = fallback(tmp_path, {"reason": "no_step_clears"})
    assert out["_degraded"] == "llm_summariser_unavailable"


def test_fallback_carries_reason_and_trajectory_tail(tmp_path):
    _write_trajectory(tmp_path, [{"message": "tail " + "x" * 900}])
    out = fallback(tmp_path, {"reason": "no_step_clears"})
    assert out["failure_cause"] == "no_step_clears"
    assert len(out["mechanism"]) == FIELD_CAPS["mechanism"]
    assert out["hypothesis"] == out["hyperparameters"] == ""
    assert out["measurements"] == out["do_differently"] == ""


def test_fallback_caps_a_long_reason(tmp_path):
    out = fallback(tmp_path, {"reason": "r" * 1000})
    assert len(out["failure_cause"]) == FIELD_CAPS["failure_cause"]


def test_fallback_without_trajectory_has_empty_mechanism(tmp_path):
    out = fallback(tmp_path, {})
    assert out["mechanism"] == ""
    assert out["failure_cause"] == ""


# --- summarize_iteration ------------------------------------------------


FACTS = {"reward": 0.0, "reason": "no_step_clears", "graded_step": 1750, "n_seeds": 2}


def test_summarize_iteration_happy_path(tmp_path, monkeypatch):
    _write_trajectory(tmp_path, [{"message": "account " + "x" * 300}])
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data)
        return _FakeResponse(_reply(VALID_JSON))

    monkeypatch.setattr(summariser.urllib.request, "urlopen", fake_urlopen)

    out = summarize_iteration(tmp_path, FACTS)
    assert tuple(out) == SCHEMA_KEYS
    assert "_degraded" not in out
    assert out["hyperparameters"] == "lr=0.05, momentum=0.95, ns_steps=5"

    # verifier facts are handed over as settled, and the agent's account is included
    prompt = seen["body"]["messages"][0]["content"]
    assert "do not dispute or re-judge these" in prompt
    assert '"graded_step": 1750' in prompt
    assert '"n_seeds": 2' in prompt
    assert "account " in prompt


def test_summarize_iteration_survives_a_dead_summariser(tmp_path, monkeypatch):
    _write_trajectory(tmp_path, [{"message": "account " + "x" * 300}])

    def fake_urlopen(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(summariser.urllib.request, "urlopen", fake_urlopen)

    out = summarize_iteration(tmp_path, FACTS)  # must not raise
    assert out["_degraded"].startswith("summariser_failed")
    assert out["failure_cause"] == "no_step_clears"


def test_summarize_iteration_degrades_on_off_schema_reply(tmp_path, monkeypatch):
    _write_trajectory(tmp_path, [{"message": "account " + "x" * 300}])
    monkeypatch.setattr(
        summariser.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResponse(_reply("I decline to answer.")))
    out = summarize_iteration(tmp_path, FACTS)
    assert out["_degraded"].startswith("summariser_failed")


def test_summarize_iteration_without_trajectory_never_calls_the_bridge(tmp_path):
    # the autouse _no_network blocker would raise if the bridge were touched
    out = summarize_iteration(tmp_path, FACTS)
    assert out["_degraded"] == "llm_summariser_unavailable"


# --- render -------------------------------------------------------------


def test_render_empty_summary():
    assert render({}) == ""
    assert render(None) == ""


def test_render_degraded_summary_is_labelled(tmp_path):
    out = render({"mechanism": "last thing it said", "_degraded": "llm_summariser_unavailable"})
    assert out.startswith("_Summary degraded")
    assert "llm_summariser_unavailable" in out
    assert "last thing it said" in out


def test_render_degraded_with_no_account():
    out = render({"mechanism": "", "_degraded": "summariser_failed: boom"})
    assert out.startswith("_Summary degraded")
    assert "(no account recovered)" in out


def test_render_full_summary_is_ordered_bullets():
    out = render(_coerce(VALID_JSON))
    assert "- **Hypothesis:**" in out
    for label in ("Mechanism", "Hyperparameters", "Measured",
                  "Why it did not score higher", "Next attempt should change"):
        assert f"- **{label}:**" in out
    assert out.index("Hypothesis") < out.index("Mechanism") < out.index("Measured")


def test_render_skips_empty_fields():
    out = render({"hypothesis": "h", "mechanism": "", "hyperparameters": "",
                  "measurements": "", "failure_cause": "", "do_differently": ""})
    assert out == "- **Hypothesis:** h"
