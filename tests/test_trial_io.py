"""Trial-result parser tests, asserted against a REAL captured trial.

`tests/fixtures/track3_trial/` is a verbatim copy of one completed agentic trial
from the Track-3 production pipeline (reward 0.5, reason `graded_step=3200`, two
graded seeds). Every assertion about the happy path below reads that fixture
rather than a hand-built mock, so a parser that only satisfies invented shapes
cannot pass. The fixture is immutable — `tests/test_track3_fixture.py` guards it
— so tests that need to write into a trial directory build one under `tmp_path`.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest

from track3.trial_io import (
    _budget_used,
    agent_findings,
    find_trial,
    parent_artifacts,
    read_trial,
    task_budget_hours,
    task_budget_secs,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "track3_trial"
HARNESS_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = HARNESS_ROOT / "tasks" / "2739a678-1759-516d-8ba7-1cd023267ea8"

EXPECTED_KEYS = {
    "reward",
    "reason",
    "graded_step",
    "n_seeds",
    "outcome",
    "harness_error",
    "budget_used_frac",
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_tokens",
    "cost_usd",
    "started_at",
    "finished_at",
    "trial_dir",
}


def _write_trial(root: Path, *, started: str, finished: str, timeout_sec=None) -> Path:
    """A minimal trial dir with only the fields `_budget_used` reads."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "verifier").mkdir(exist_ok=True)
    (root / "verifier" / "reward_full.json").write_text(
        json.dumps({"reward": 0.0, "reason": ""})
    )
    (root / "result.json").write_text(
        json.dumps({"started_at": started, "finished_at": finished})
    )
    if timeout_sec is not None:
        (root / "config.json").write_text(
            json.dumps({"agent": {"timeout_sec": timeout_sec}})
        )
    return root


# --------------------------------------------------------------------------
# read_trial against the real trial
# --------------------------------------------------------------------------


def test_read_trial_returns_all_documented_keys():
    row = read_trial(FIXTURE)
    assert EXPECTED_KEYS <= set(row), f"missing keys: {EXPECTED_KEYS - set(row)}"


def test_read_trial_reads_reward_full_not_reward_json():
    """reward.json is a sibling judge module's input; the parser reads reward_full."""
    row = read_trial(FIXTURE)
    assert row["reward"] == 0.5
    assert row["reason"] == "graded_step=3200"


def test_read_trial_derives_graded_step_and_outcome():
    row = read_trial(FIXTURE)
    assert row["graded_step"] == 3200
    assert row["outcome"] == "graded_pass"


def test_read_trial_counts_seeds_from_logs_when_metrics_absent():
    row = read_trial(FIXTURE)
    assert row["n_seeds"] == 2


def test_read_trial_carries_agent_result_fields():
    res = json.loads((FIXTURE / "result.json").read_text())
    agent = res["agent_result"]
    row = read_trial(FIXTURE)
    assert row["n_input_tokens"] == agent["n_input_tokens"]
    assert row["n_output_tokens"] == agent["n_output_tokens"]
    assert row["n_cache_tokens"] == agent["n_cache_tokens"]
    assert row["cost_usd"] == agent["cost_usd"]
    assert row["started_at"] == res["started_at"]
    assert row["finished_at"] == res["finished_at"]
    assert row["trial_dir"] == str(FIXTURE)
    assert isinstance(row["trial_dir"], str)


def test_read_trial_reports_no_harness_error_for_clean_trial():
    """This trial's own config.json records no `agent.timeout_sec`, so the fraction is
    measured against the 18000 s default — and the run overshoots it, which is exactly
    why the denominator must come from the trial rather than from today's task.toml."""
    res = json.loads((FIXTURE / "result.json").read_text())
    span = (
        datetime.fromisoformat(res["finished_at"].replace("Z", "+00:00"))
        - datetime.fromisoformat(res["started_at"].replace("Z", "+00:00"))
    ).total_seconds()
    row = read_trial(FIXTURE)
    assert row["harness_error"] is False
    assert row["budget_used_frac"] == pytest.approx(span / 18000.0)
    assert row["budget_used_frac"] > 0.0


def test_read_trial_merges_parent_artifacts():
    row = read_trial(FIXTURE)
    assert row["parent_source"] == parent_artifacts(FIXTURE)["parent_source"]
    assert "parent_curve" in row


def test_read_trial_on_empty_dir_does_not_raise(tmp_path):
    row = read_trial(tmp_path)
    assert row["reward"] == 0.0
    assert row["reason"] == ""
    assert row["graded_step"] is None
    assert row["n_seeds"] == 0
    assert row["harness_error"] is False
    assert row["budget_used_frac"] is None
    assert EXPECTED_KEYS <= set(row)


def test_read_trial_survives_corrupt_json(tmp_path):
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier" / "reward_full.json").write_text("{not json")
    (tmp_path / "result.json").write_text("]]]")
    row = read_trial(tmp_path)
    assert row["reward"] == 0.0


# --------------------------------------------------------------------------
# parent_artifacts
# --------------------------------------------------------------------------


def test_parent_source_is_read_from_the_real_submission():
    out = parent_artifacts(FIXTURE)
    assert out["parent_source"].strip(), "parent_source is empty"
    assert out["parent_source_lines"] > 1
    assert out["parent_source_truncated"] is False


def test_parent_curve_has_both_seeds():
    curve = parent_artifacts(FIXTURE)["parent_curve"]
    assert set(curve) == {"0", "1"}


def test_parent_curve_is_thinned_to_at_most_thirteen_points():
    curve = parent_artifacts(FIXTURE)["parent_curve"]
    for seed, pts in curve.items():
        assert len(pts) <= 13, f"seed {seed} kept {len(pts)} points"
        assert pts, f"seed {seed} is empty"


def test_parent_curve_points_are_int_float_and_strictly_increasing():
    curve = parent_artifacts(FIXTURE)["parent_curve"]
    for seed, pts in curve.items():
        steps = []
        for point in pts:
            assert len(point) == 2, f"seed {seed} point {point} is not a pair"
            step, val = point
            assert isinstance(step, int) and not isinstance(step, bool)
            assert isinstance(val, float)
            steps.append(step)
        assert steps == sorted(steps), f"seed {seed} steps out of order"
        assert len(set(steps)) == len(steps), f"seed {seed} has duplicate steps"


def test_parent_curve_keeps_the_final_point():
    import re

    curve = parent_artifacts(FIXTURE)["parent_curve"]
    for log in FIXTURE.glob("artifacts/**/full_seed*.log"):
        seed = log.stem.replace("full_seed", "")
        pts = re.findall(
            r"step:(\d+)/\d+ val_loss:([\d.]+)", log.read_text(errors="replace")
        )
        last = (int(pts[-1][0]), float(pts[-1][1]))
        assert tuple(curve[seed][-1]) == last


def test_parent_artifacts_prefers_the_shallowest_optimizer(tmp_path):
    deep = tmp_path / "artifacts" / "a" / "b" / "c" / "optimizer.py"
    shallow = tmp_path / "artifacts" / "workspace" / "optimizer.py"
    deep.parent.mkdir(parents=True)
    shallow.parent.mkdir(parents=True)
    deep.write_text("# DEEP\n")
    shallow.write_text("# SHALLOW\n")
    out = parent_artifacts(tmp_path)
    assert out["parent_source"] == "# SHALLOW\n"


def test_parent_artifacts_returns_empty_dict_when_nothing_found(tmp_path):
    out = parent_artifacts(tmp_path)
    assert out == {}
    assert "parent_source" not in out
    assert "parent_curve" not in out


def test_parent_artifacts_omits_curve_when_logs_have_no_val_loss(tmp_path):
    log = tmp_path / "artifacts" / "logs" / "full_seed0.log"
    log.parent.mkdir(parents=True)
    log.write_text("nothing parseable here\n")
    assert "parent_curve" not in parent_artifacts(tmp_path)


def test_parent_source_truncates_at_cap(tmp_path):
    src = tmp_path / "artifacts" / "optimizer.py"
    src.parent.mkdir(parents=True)
    src.write_text("x\n" * 20000)
    out = parent_artifacts(tmp_path)
    assert len(out["parent_source"]) == 18000
    assert out["parent_source_truncated"] is True


# --------------------------------------------------------------------------
# agent_findings
# --------------------------------------------------------------------------


def test_agent_findings_returns_bounded_str_from_real_trajectory():
    found = agent_findings(FIXTURE)
    assert isinstance(found, str)
    assert 0 < len(found) <= 1200
    # Stripped BEFORE truncation, per the source: the cap may land mid-sentence and
    # leave a trailing space, but no leading whitespace ever survives.
    assert found == found.lstrip()
    assert found.startswith(found.split("\n", 1)[0].strip())


def test_agent_findings_takes_the_last_substantive_message(tmp_path):
    (tmp_path / "agent").mkdir()
    long_a, long_b = "A" * 300, "B" * 300
    (tmp_path / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"message": long_a},
                    {"message": long_b},
                    {"message": "too short"},
                    {"message": None},
                ]
            }
        )
    )
    assert agent_findings(tmp_path) == long_b


def test_agent_findings_respects_cap(tmp_path):
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "trajectory.json").write_text(
        json.dumps({"steps": [{"message": "Z" * 5000}]})
    )
    assert len(agent_findings(tmp_path, cap=300)) == 300


def test_agent_findings_empty_when_missing_or_broken(tmp_path):
    assert agent_findings(tmp_path) == ""
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "trajectory.json").write_text("{broken")
    assert agent_findings(tmp_path) == ""
    (tmp_path / "agent" / "trajectory.json").write_text(json.dumps({"steps": []}))
    assert agent_findings(tmp_path) == ""


# --------------------------------------------------------------------------
# _budget_used
# --------------------------------------------------------------------------


def test_budget_used_denominator_comes_from_trial_config(tmp_path):
    trial = _write_trial(
        tmp_path / "t",
        started="2026-08-14T22:00:00Z",
        finished="2026-08-14T22:48:00Z",  # 2880 s
        timeout_sec=28800,
    )
    res = json.loads((trial / "result.json").read_text())
    assert _budget_used(res, trial) == pytest.approx(0.1)
    assert read_trial(trial)["budget_used_frac"] == pytest.approx(0.1)


def test_budget_used_falls_back_to_18000_without_config(tmp_path):
    trial = _write_trial(
        tmp_path / "t",
        started="2026-08-14T22:00:00Z",
        finished="2026-08-14T22:30:00Z",  # 1800 s
    )
    assert not (trial / "config.json").exists()
    res = json.loads((trial / "result.json").read_text())
    assert _budget_used(res, trial) == pytest.approx(1800.0 / 18000.0)


def test_budget_used_ignores_unreadable_config(tmp_path):
    trial = _write_trial(
        tmp_path / "t",
        started="2026-08-14T22:00:00Z",
        finished="2026-08-14T22:30:00Z",
    )
    (trial / "config.json").write_text("{not json")
    res = json.loads((trial / "result.json").read_text())
    assert _budget_used(res, trial) == pytest.approx(1800.0 / 18000.0)


def test_budget_used_is_none_when_timestamps_unusable(tmp_path):
    assert _budget_used({}, tmp_path) is None
    assert _budget_used({"started_at": "nonsense", "finished_at": None}, tmp_path) is None


# --------------------------------------------------------------------------
# find_trial
# --------------------------------------------------------------------------


def _job_with_trials(root: Path, names_and_mtimes) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, mtime in names_and_mtimes:
        d = root / name
        d.mkdir()
        (d / "result.json").write_text("{}")
        os.utime(d, (mtime, mtime))
    return root


def test_find_trial_picks_the_newest_child_with_result_json(tmp_path):
    now = time.time()
    job = _job_with_trials(
        tmp_path / "job", [("old", now - 100), ("new", now), ("noresult", now)]
    )
    (job / "noresult" / "result.json").unlink()
    os.utime(job / "noresult", (now + 50, now + 50))
    assert find_trial(job, now - 200) == job / "new"


def test_find_trial_returns_none_for_empty_dir(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    assert find_trial(job, time.time()) is None


def test_find_trial_ignores_children_without_result_json(tmp_path):
    job = tmp_path / "job"
    (job / "bare").mkdir(parents=True)
    assert find_trial(job, 0.0) is None


def test_find_trial_refalls_back_when_all_candidates_predate_since(tmp_path):
    """The re-glob fallback: a stale-but-real trial beats returning nothing."""
    now = time.time()
    job = _job_with_trials(
        tmp_path / "job", [("older", now - 9000), ("newer", now - 6000)]
    )
    assert find_trial(job, now) == job / "newer"


def test_find_trial_honours_the_five_second_grace(tmp_path):
    now = time.time()
    job = _job_with_trials(tmp_path / "job", [("a", now - 3)])
    assert find_trial(job, now) == job / "a"


# --------------------------------------------------------------------------
# task budget helpers (task_dir is a PARAMETER, not a module global)
# --------------------------------------------------------------------------


def test_task_budget_reads_the_real_task_toml():
    assert task_budget_secs(TASK_DIR) == pytest.approx(28800.0)
    assert task_budget_hours(TASK_DIR) == "8 hours of wall clock"


def test_task_budget_is_none_without_task_toml(tmp_path):
    assert task_budget_secs(tmp_path) is None
    assert task_budget_hours(tmp_path) == "as stated in the task instruction"


def test_task_budget_hours_formats_fractional_hours(tmp_path):
    (tmp_path / "task.toml").write_text("[agent]\ntimeout_sec = 5400.0\n")
    assert task_budget_hours(tmp_path) == "1.5 hours of wall clock"


def test_task_budget_secs_none_on_malformed_toml(tmp_path):
    (tmp_path / "task.toml").write_text("[agent\n")
    assert task_budget_secs(tmp_path) is None
