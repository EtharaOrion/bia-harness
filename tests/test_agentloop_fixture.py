"""Completeness checks for the real-trial Track-3 fixture.

`tests/fixtures/track3_trial/` is a verbatim (partially truncated) copy of one
completed agentic trial from the Track-3 production pipeline. Downstream trial
parsers are unit-tested against it instead of hand-written mocks, so these tests
guard the fixture itself: if a file goes missing or gets mangled, the parser
tests would fail for the wrong reason.
"""

import json
import re
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "track3_trial"

REQUIRED_FILES = [
    "verifier/reward_full.json",
    "verifier/reward.json",
    "result.json",
    "config.json",
    "agent/trajectory.json",
    "agent/claude-code.txt",
]

VAL_LOSS_RE = re.compile(r"step:(\d+)/\d+ val_loss:([\d.]+)")

MAX_FIXTURE_BYTES = 2 * 1024 * 1024  # 2 MB


@pytest.mark.parametrize("rel", REQUIRED_FILES)
def test_required_file_exists(rel):
    path = FIXTURE / rel
    assert path.is_file(), f"missing fixture file: {rel}"
    assert path.stat().st_size > 0, f"empty fixture file: {rel}"


def test_reward_full_has_numeric_reward_and_reason():
    data = json.loads((FIXTURE / "verifier" / "reward_full.json").read_text())
    assert "reward" in data, "reward_full.json has no 'reward' key"
    assert isinstance(data["reward"], (int, float)), (
        f"reward is {type(data['reward']).__name__}, expected numeric"
    )
    assert not isinstance(data["reward"], bool), "reward must not be a bool"
    assert "reason" in data, "reward_full.json has no 'reason' key"
    assert str(data["reason"]).strip(), "reward_full.json 'reason' is empty"


def test_reward_json_parses():
    data = json.loads((FIXTURE / "verifier" / "reward.json").read_text())
    assert isinstance(data, dict)


def test_result_and_config_parse():
    assert isinstance(json.loads((FIXTURE / "result.json").read_text()), dict)
    assert isinstance(json.loads((FIXTURE / "config.json").read_text()), dict)


def test_trajectory_json_has_steps_list():
    data = json.loads((FIXTURE / "agent" / "trajectory.json").read_text())
    assert isinstance(data, dict), "trajectory.json top level must be an object"
    assert "steps" in data, "trajectory.json has no 'steps' key"
    assert isinstance(data["steps"], list), "trajectory.json 'steps' must be a list"
    assert data["steps"], "trajectory.json 'steps' is empty"


def test_claude_code_txt_has_parseable_jsonl_lines():
    text = (FIXTURE / "agent" / "claude-code.txt").read_text(errors="replace")
    parsed = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        parsed += 1
    assert parsed >= 1, "claude-code.txt has no JSON-parseable lines"


def test_at_least_one_optimizer_py_under_artifacts():
    hits = list(FIXTURE.glob("artifacts/**/optimizer.py"))
    assert hits, "no optimizer.py found under artifacts/"
    assert any(p.stat().st_size > 0 for p in hits), "all optimizer.py files are empty"


def test_full_seed_log_has_val_loss_lines():
    logs = list(FIXTURE.glob("artifacts/**/full_seed*.log"))
    assert logs, "no full_seed*.log found under artifacts/"
    best = 0
    for log in logs:
        matches = VAL_LOSS_RE.findall(log.read_text(errors="replace"))
        best = max(best, len(matches))
    assert best >= 2, (
        f"no full_seed*.log has >=2 'step:N/M val_loss:X' lines (max found: {best})"
    )


def test_fixture_total_size_under_limit():
    total = sum(p.stat().st_size for p in FIXTURE.rglob("*") if p.is_file())
    assert total < MAX_FIXTURE_BYTES, (
        f"fixture is {total} bytes, limit is {MAX_FIXTURE_BYTES}"
    )
