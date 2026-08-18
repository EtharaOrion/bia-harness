"""Unit tests for the veto-only trajectory judge (``runner/agentloop/judge.py``).

The judge is a VETO: it may multiply a reward by 0 or 1 and can never raise one.
These tests pin the properties that keep it that way, plus the two load-bearing
``SystemExit`` sites (unreachable bridge, empty input) that an upstream caller
deliberately catches via ``BaseException``.

NO LIVE NETWORK. Every test runs behind two guards installed by the autouse
``_no_network`` fixture: ``socket.socket.connect`` is poisoned at the OS-socket
level, and ``urllib.request.urlopen`` defaults to raising. Tests that need a
reply install their own fake ``urlopen`` on top; monkeypatch unwinds LIFO, so
the guards are always restored.
"""

import json
import re
import socket
import time
import urllib.request
from pathlib import Path

import pytest

from agentloop.judge import (
    BRIDGE,
    _agent_transcript,
    _parse_verdict,
    _submitted_optimizer,
    _verifier_reward,
    ask,
    grade_attempt,
    load_rubrics,
)

SEP = "\n\n---\n\n"


# --------------------------------------------------------------------------
# no-network guards
# --------------------------------------------------------------------------


class NetworkAccessAttempted(AssertionError):
    """Raised if any test tries to touch a real socket or a real urlopen."""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Poison every route to the network for the whole module."""

    def _blocked_connect(self, *a, **k):
        raise NetworkAccessAttempted("test attempted a real socket connect")

    def _blocked_urlopen(*a, **k):
        raise NetworkAccessAttempted("test attempted a real urlopen")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(urllib.request, "urlopen", _blocked_urlopen)


class _FakeResponse:
    """Minimal stand-in for the object ``urlopen`` returns."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_urlopen(monkeypatch, texts):
    """Serve ``texts`` (one per call) as bridge replies; record every call."""
    calls = []
    seq = list(texts)

    def fake_urlopen(req, timeout=None, **kw):
        calls.append({"req": req, "timeout": timeout})
        text = seq[len(calls) - 1] if len(calls) <= len(seq) else seq[-1]
        return _FakeResponse({"content": [{"type": "text", "text": text}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def _write_trial(root: Path, *, transcript_text=None, optimizer_src=None, reward=None):
    """Build a minimal trial directory of the shape the judge reads."""
    root.mkdir(parents=True, exist_ok=True)
    if transcript_text is not None:
        (root / "agent").mkdir(exist_ok=True)
        rec = {"type": "assistant", "message": {"content": [{"type": "text", "text": transcript_text}]}}
        (root / "agent" / "claude-code.txt").write_text(json.dumps(rec) + "\n")
    if optimizer_src is not None:
        d = root / "artifacts" / "workspace" / "submission"
        d.mkdir(parents=True, exist_ok=True)
        (d / "optimizer.py").write_text(optimizer_src)
    if reward is not None:
        (root / "verifier").mkdir(exist_ok=True)
        (root / "verifier" / "reward.json").write_text(json.dumps(reward))
    return root


# --------------------------------------------------------------------------
# the two load-bearing SystemExit sites
# --------------------------------------------------------------------------


def test_grade_attempt_raises_system_exit_on_empty_input(tmp_path):
    """No transcript AND no optimizer => refuse to grade, as SystemExit.

    Not ValueError, not RuntimeError: an upstream caller catches BaseException
    precisely because this site (and ``ask``'s) are SystemExit.
    """
    trial = tmp_path / "trial"
    trial.mkdir()

    with pytest.raises(SystemExit) as ei:
        grade_attempt(trial)

    assert "refusing" in str(ei.value).lower()
    assert not (trial / "rubric_verdicts.json").exists(), (
        "must not emit a verdict file when refusing to grade"
    )


def test_grade_attempt_empty_input_never_reaches_the_bridge(tmp_path):
    """The refusal happens before any LLM call (the _no_network guard proves it)."""
    trial = tmp_path / "trial"
    trial.mkdir()
    with pytest.raises(SystemExit):
        grade_attempt(trial)


def test_ask_raises_system_exit_when_bridge_never_answers(monkeypatch):
    """Exhausted retries => SystemExit('judge unreachable'), not a plain Exception."""
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    attempts = []

    def always_fails(req, timeout=None, **kw):
        attempts.append(timeout)
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", always_fails)

    with pytest.raises(SystemExit) as ei:
        ask("claude-opus-5", "hello")

    assert "judge unreachable" in str(ei.value)
    assert len(attempts) == 5, f"default retries must be 5, saw {len(attempts)}"
    assert slept == [8, 8, 8, 8, 8], f"must back off 8s between retries, saw {slept}"


def test_ask_is_not_a_normal_exception(monkeypatch):
    """SystemExit must not be catchable as Exception, or upstream logic changes."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))
    )
    try:
        ask("m", "p")
    except Exception:  # noqa: BLE001 - deliberately proving it is NOT an Exception
        pytest.fail("ask raised Exception; the SystemExit site was softened")
    except SystemExit:
        pass


# --------------------------------------------------------------------------
# transport shape: bridge URL, timeout, headers
# --------------------------------------------------------------------------


def test_bridge_default_is_host_loopback(monkeypatch):
    assert BRIDGE == "http://127.0.0.1:8765/v1/messages"


def test_ask_uses_1800s_timeout_and_returns_concatenated_text(monkeypatch):
    calls = _install_urlopen(monkeypatch, ["verdict text"])
    out = ask("claude-opus-5", "prompt")
    assert out == "verdict text"
    assert calls[0]["timeout"] == 1800, "urlopen timeout must be 1800s"


def test_ask_sends_no_api_key_header(monkeypatch):
    """Deliberate asymmetry with the sibling summariser: the judge sends NO x-api-key."""
    calls = _install_urlopen(monkeypatch, ["ok"])
    ask("claude-opus-5", "prompt")

    req = calls[0]["req"]
    keys = {k.lower() for k in req.headers}
    assert keys == {"content-type", "anthropic-version"}, (
        f"judge must send exactly Content-Type + anthropic-version, got {sorted(keys)}"
    )
    assert not req.has_header("X-api-key"), "judge must NOT send an x-api-key header"
    assert req.get_header("Anthropic-version") == "2023-06-01"
    assert req.get_header("Content-type") == "application/json"
    assert req.full_url == BRIDGE


# --------------------------------------------------------------------------
# _parse_verdict
# --------------------------------------------------------------------------


def test_parse_verdict_recovers_from_json_fence():
    raw = 'Here is my audit.\n```json\n{"overall_pass": false, "verdicts": {}}\n```\nDone.'
    out = _parse_verdict(raw)
    assert out == {"overall_pass": False, "verdicts": {}}


def test_parse_verdict_recovers_bare_object_with_trailing_prose():
    """A greedy first-{-to-last-} span breaks when prose after the object has a brace."""
    raw = (
        '{"overall_pass": true, "summary": "clean run"}\n\n'
        "I hope that helps. (Formatting note: use {curly} braces sparingly.)"
    )
    assert _parse_verdict(raw) == {"overall_pass": True, "summary": "clean run"}

    greedy = re.search(r"\{.*\}", raw, re.S)
    with pytest.raises(json.JSONDecodeError):
        json.loads(greedy.group(0))


def test_parse_verdict_handles_closing_brace_inside_a_string_value():
    """The quote-aware balancer exists for exactly this: a '}' inside evidence."""
    raw = (
        '{"verdicts": {"design_rationale_present": '
        '{"pass": false, "evidence": "agent wrote: if cond: pass}  # stray"}}, '
        '"overall_pass": false}\n'
        "Trailing commentary with an unbalanced } brace."
    )
    out = _parse_verdict(raw)
    assert out is not None, "quote-aware balancer failed to recover the verdict"
    assert out["overall_pass"] is False
    assert "stray" in out["verdicts"]["design_rationale_present"]["evidence"]

    greedy = re.search(r"\{.*\}", raw, re.S)
    with pytest.raises(json.JSONDecodeError):
        json.loads(greedy.group(0))


def test_parse_verdict_returns_none_without_json():
    assert _parse_verdict("I decline to answer in JSON.") is None
    assert _parse_verdict("") is None
    assert _parse_verdict("{not: valid, json]") is None


def test_parse_verdict_rejects_non_dict_json():
    assert _parse_verdict("[1, 2, 3]") is None


# --------------------------------------------------------------------------
# _agent_transcript
# --------------------------------------------------------------------------


def test_agent_transcript_keeps_only_long_assistant_text_blocks(tmp_path):
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)

    keep_a = "A" * 250
    keep_c = "C" * 300
    lines = [
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": keep_a}]}}),
        "",
        "{ not json at all",
        json.dumps({"type": "user", "message": {"content": [{"type": "text", "text": "U" * 400}]}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "B" * 100}]}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "D" * 200}]}}),
        json.dumps({"type": "assistant", "message": {"content": "a bare string, not a list"}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "text": "T" * 400}]}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": keep_c}]}}),
    ]
    (trial / "agent" / "claude-code.txt").write_text("\n".join(lines) + "\n")

    out = _agent_transcript(trial)
    assert out == keep_a + SEP + keep_c
    assert "B" * 100 not in out
    assert "D" * 200 not in out, "len == 200 is not > 200; must be dropped"
    assert "U" * 400 not in out, "non-assistant entries must be dropped"
    assert "T" * 400 not in out, "non-text content items must be dropped"


def test_agent_transcript_missing_file_is_empty_string(tmp_path):
    assert _agent_transcript(tmp_path / "nope") == ""


def test_agent_transcript_no_qualifying_blocks_is_empty_string(tmp_path):
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "claude-code.txt").write_text(
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "short"}]}})
    )
    assert _agent_transcript(trial) == ""


def test_agent_transcript_truncates_head_and_tail(tmp_path):
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    body = "X" * 5000
    (trial / "agent" / "claude-code.txt").write_text(
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": body}]}})
    )

    limit = 1000
    out = _agent_transcript(trial, limit=limit)
    marker = "\n\n[... transcript truncated ...]\n\n"
    assert marker in out
    head = limit * 2 // 3
    assert out == body[:head] + marker + body[-(limit - head):]


# --------------------------------------------------------------------------
# _submitted_optimizer
# --------------------------------------------------------------------------


def test_submitted_optimizer_picks_the_shallowest(tmp_path):
    trial = tmp_path / "trial"
    shallow = trial / "artifacts"
    deep = trial / "artifacts" / "workspace" / "submission"
    shallow.mkdir(parents=True)
    deep.mkdir(parents=True)
    (shallow / "optimizer.py").write_text("# SHALLOW\n")
    (deep / "optimizer.py").write_text("# DEEP\n")

    assert _submitted_optimizer(trial) == "# SHALLOW\n"


def test_submitted_optimizer_missing_is_empty_string(tmp_path):
    assert _submitted_optimizer(tmp_path) == ""


def test_submitted_optimizer_truncates_to_limit(tmp_path):
    trial = tmp_path / "trial"
    d = trial / "artifacts"
    d.mkdir(parents=True)
    (d / "optimizer.py").write_text("z" * 500)
    assert _submitted_optimizer(trial, limit=100) == "z" * 100


# --------------------------------------------------------------------------
# _verifier_reward
# --------------------------------------------------------------------------


def test_verifier_reward_reads_reward_json_not_reward_full(tmp_path):
    """reward.json is the file. A sibling module reads reward_full.json; not this one."""
    trial = tmp_path / "trial"
    (trial / "verifier").mkdir(parents=True)
    (trial / "verifier" / "reward.json").write_text(json.dumps({"reward": 0.5, "reason": "ok"}))
    (trial / "verifier" / "reward_full.json").write_text(json.dumps({"reward": 9.9, "reason": "WRONG FILE"}))

    out = _verifier_reward(trial)
    assert out == {"reward": 0.5, "reason": "ok"}
    assert out["reason"] != "WRONG FILE"


def test_verifier_reward_missing_is_empty_dict(tmp_path):
    assert _verifier_reward(tmp_path) == {}


# --------------------------------------------------------------------------
# load_rubrics / _rubrics_path
# --------------------------------------------------------------------------


def test_load_rubrics_from_explicit_path(tmp_path):
    p = tmp_path / "rubrics.jsonl"
    p.write_text(
        json.dumps({"id": "r1", "rubric": "no hacking"})
        + "\n\n"
        + json.dumps({"id": "r2", "rubric": "honest 8-hour budget"})
        + "\n"
    )
    out = load_rubrics(p)
    assert [r["id"] for r in out] == ["r1", "r2"]
    assert all("rubric" in r for r in out)


def test_load_rubrics_resolves_this_repos_task_layout():
    """tasks/<uuid>/tests/rubrics.jsonl must be discoverable with no argument."""
    out = load_rubrics()
    assert out, "no rubrics resolved for this repo layout"
    assert all("id" in r and "rubric" in r for r in out)


def test_importing_the_module_does_not_require_rubrics(tmp_path):
    """No import-time resolution: a missing rubrics file must not break collection.

    Imports a copy of judge.py planted where no rubrics.jsonl is resolvable (no file
    beside it, no tasks/*/tests/ tree above it). Import must succeed anyway; only the
    call must fail. Resolving at module scope would turn a missing rubrics file into a
    collection error for every test that imports this module.
    """
    import importlib.util
    import shutil

    import agentloop.judge as real

    planted = tmp_path / "isolated" / "runner" / "agentloop"
    planted.mkdir(parents=True)
    shutil.copy(real.__file__, planted / "judge.py")
    assert not list(tmp_path.glob("**/rubrics.jsonl")), "sandbox must contain no rubrics"

    spec = importlib.util.spec_from_file_location("judge_isolated", planted / "judge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # raises here if resolution happens at import time

    with pytest.raises(FileNotFoundError):
        mod._rubrics_path()
    with pytest.raises(FileNotFoundError):
        mod.load_rubrics()


# --------------------------------------------------------------------------
# grade_attempt end to end (mocked bridge)
# --------------------------------------------------------------------------


def test_grade_attempt_happy_path_writes_rubric_verdicts_json(tmp_path, monkeypatch):
    trial = _write_trial(
        tmp_path / "trial",
        transcript_text="I considered the 8-hour budget carefully. " + "R" * 300,
        optimizer_src="def build_optimizer(params):\n    return None\n",
        reward={"reward": 0.0, "reason": "no_step_clears"},
    )

    verdict = json.dumps(
        {
            "verdicts": {"design_rationale_present": {"pass": True, "evidence": "none observed"}},
            "overall_pass": True,
            "summary": "no rubric violation observed",
        }
    )
    calls = _install_urlopen(monkeypatch, [verdict])

    out = grade_attempt(trial)

    assert len(calls) == 1, "a parseable first reply must not be re-asked"
    assert out["_veto_only"] is True
    assert out["overall_pass"] is True
    assert out["_judge_model"] == "claude-opus-5"
    assert out.get("_indeterminate") is None
    assert "_note" in out and "never raise" in out["_note"]
    assert out["_rubrics_path"] and Path(out["_rubrics_path"]).is_file()

    written = trial / "rubric_verdicts.json"
    assert written.is_file(), "grade_attempt must write <trial>/rubric_verdicts.json"
    on_disk = json.loads(written.read_text())
    assert on_disk == out
    assert on_disk["_veto_only"] is True


def test_grade_attempt_prompt_carries_rubrics_transcript_and_optimizer(tmp_path, monkeypatch):
    trial = _write_trial(
        tmp_path / "trial",
        transcript_text="MARKER_TRANSCRIPT " + "R" * 300,
        optimizer_src="# MARKER_OPTIMIZER\n",
        reward={"reward": 0.0, "reason": "MARKER_REASON"},
    )
    calls = _install_urlopen(monkeypatch, ['{"overall_pass": true, "verdicts": {}}'])

    grade_attempt(trial)

    body = json.loads(calls[0]["req"].data.decode())
    prompt = body["messages"][0]["content"]
    assert "MARKER_TRANSCRIPT" in prompt
    assert "MARKER_OPTIMIZER" in prompt
    assert "MARKER_REASON" in prompt
    assert "VETO" in prompt and "never improve its score" in prompt
    for r in load_rubrics():
        assert f"[{r['id']}]" in prompt, f"rubric {r['id']} missing from prompt"


def test_grade_attempt_budget_hours_extracted_from_rubrics(tmp_path, monkeypatch):
    trial = _write_trial(tmp_path / "trial", optimizer_src="x = 1\n")
    _install_urlopen(monkeypatch, ['{"overall_pass": true, "verdicts": {}}'])
    out = grade_attempt(trial)

    rubric_text = " ".join(r["rubric"] for r in load_rubrics())
    expected = re.search(r"(\d+)-hour", rubric_text)
    assert out["_budget_hours_in_rubrics"] == (expected.group(1) if expected else None)


def test_grade_attempt_indeterminate_after_three_unparseable_replies(tmp_path, monkeypatch):
    trial = _write_trial(tmp_path / "trial", optimizer_src="x = 1\n")
    calls = _install_urlopen(monkeypatch, ["I refuse to reply in JSON."] * 3)

    out = grade_attempt(trial)

    assert len(calls) == 3, f"must try exactly 3 times, saw {len(calls)}"
    assert out["_indeterminate"] is True
    assert out["overall_pass"] is None
    assert out["verdicts"] == {}
    assert "human review" in out["summary"]
    assert out["_veto_only"] is True
    assert json.loads((trial / "rubric_verdicts.json").read_text())["_indeterminate"] is True


def test_grade_attempt_retries_until_parseable(tmp_path, monkeypatch):
    trial = _write_trial(tmp_path / "trial", optimizer_src="x = 1\n")
    calls = _install_urlopen(
        monkeypatch, ["prose only", '{"overall_pass": false, "verdicts": {}, "summary": "veto"}']
    )

    out = grade_attempt(trial)

    assert len(calls) == 2
    assert out["overall_pass"] is False
    assert out.get("_indeterminate") is None


def test_grade_attempt_transcript_alone_is_enough_to_grade(tmp_path, monkeypatch):
    """Only one of transcript/optimizer must be present; both empty is the refusal."""
    trial = _write_trial(tmp_path / "trial", transcript_text="T" * 400)
    _install_urlopen(monkeypatch, ['{"overall_pass": true, "verdicts": {}}'])
    assert grade_attempt(trial)["_veto_only"] is True


def test_grade_attempt_honours_model_argument(tmp_path, monkeypatch):
    trial = _write_trial(tmp_path / "trial", optimizer_src="x = 1\n")
    calls = _install_urlopen(monkeypatch, ['{"overall_pass": true, "verdicts": {}}'])

    out = grade_attempt(trial, model="claude-sonnet-4")

    assert out["_judge_model"] == "claude-sonnet-4"
    assert json.loads(calls[0]["req"].data.decode())["model"] == "claude-sonnet-4"


def test_grade_attempt_on_the_real_trial_fixture(tmp_path, monkeypatch):
    """End to end against the verbatim production trial in tests/fixtures/track3_trial.

    Copied to tmp first: grading writes rubric_verdicts.json, and the fixture is an
    immutable record guarded by tests/test_agentloop_fixture.py.
    """
    import shutil

    src = Path(__file__).resolve().parent / "fixtures" / "track3_trial"
    if not src.is_dir():
        pytest.skip("track3_trial fixture not present")

    trial = tmp_path / "track3_trial"
    shutil.copytree(src, trial)

    calls = _install_urlopen(monkeypatch, ['{"overall_pass": true, "verdicts": {}, "summary": "ok"}'])
    out = grade_attempt(trial)

    prompt = json.loads(calls[0]["req"].data.decode())["messages"][0]["content"]
    assert _agent_transcript(trial), "fixture transcript produced no assistant blocks"
    assert _submitted_optimizer(trial), "fixture yielded no optimizer source"
    assert "## The optimizer it submitted" in prompt
    assert out["_veto_only"] is True
    assert json.loads((trial / "rubric_verdicts.json").read_text())["_veto_only"] is True


def test_grade_attempt_cannot_raise_a_reward(tmp_path, monkeypatch):
    """Veto-only by construction: the verdict never carries a numeric score to add."""
    trial = _write_trial(
        tmp_path / "trial", optimizer_src="x = 1\n", reward={"reward": 0.25, "reason": "ok"}
    )
    _install_urlopen(
        monkeypatch, [json.dumps({"overall_pass": True, "verdicts": {}, "reward": 1.0, "score": 99})]
    )

    out = grade_attempt(trial)

    assert out["_veto_only"] is True
    assert _verifier_reward(trial) == {"reward": 0.25, "reason": "ok"}, (
        "the judge must not rewrite the machine-computed reward"
    )
