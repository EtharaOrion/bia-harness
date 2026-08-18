"""Trial-result parser tests, asserted against a REAL captured trial.

`tests/fixtures/track3_trial/` is a verbatim copy of one completed agentic trial
from the Track-3 production pipeline (two graded seeds; its own verifier recorded
reward 0.5, reason `graded_step=3200`). Every assertion about the happy path below
reads that fixture rather than a hand-built mock, so a parser that only satisfies
invented shapes cannot pass. The fixture is immutable — `tests/test_track3_fixture.py`
guards it — so tests that need to write into a trial directory build one under
`tmp_path`.

THE REWARD NO LONGER COMES FROM THAT VERIFIER. The verifier and the LLM judge are
UNWIRED (see `runner/track3/loop.py`); `read_trial` now derives the reward from the
parsed loss curve via `track3.reward`. The fixture's verifier reason is still read —
it is the only carrier of `reason` — but its 0.5 is not.

What the fixture's logs actually contain, measured rather than assumed:

    seed 0: 61 points, final val_loss 3.26634 — first point <= 3.28 at step 3150
    seed 1: 61 points, final val_loss 3.26717 — first point <= 3.28 at step 3175

So this run DOES reach the target, and the expected values below are computed, not
copied from the old verifier:

    step   = max(3150, 3175) = 3175      (AGREEMENT: the slowest seed sets the step)
    reward = (3500 - 3175) / 600         = 0.541666...

Those steps are read off the FULL-DENSITY 61-point logs, which is what `read_trial`
now scores. The thinned `parent_curve` retains neither 3150 nor 3175 and reports the
run as crossing at 3250 (reward 0.41666...) — a value that exists only because of the
display stride. Scoring it would have made `MAX_CURVE_POINTS` load-bearing, so the
regression tests below pin the full-density reading directly.

The residual drift from the verifier's 3200 is expected and unrelated: `grade.py`
applies a significance margin and a persistence check that `track3.reward`
deliberately does not reproduce.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest

from track3.reward import BASELINE_STEPS, TARGET_STEPS, compute_reward, reward_from_curve
from track3.trial_io import (
    _budget_used,
    agent_findings,
    find_trial,
    find_trial_with_staleness,
    parent_artifacts,
    read_trial,
    task_budget_hours,
    task_budget_secs,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "track3_trial"
HARNESS_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = HARNESS_ROOT / "tasks" / "2739a678-1759-516d-8ba7-1cd023267ea8"

# Measured from the fixture's own full-density logs, not copied from its verifier
# and not read off the thinned display curve (which would say 3250).
FIXTURE_STEP = 3175
FIXTURE_THINNED_STEP = 3250
FIXTURE_REWARD = (BASELINE_STEPS - FIXTURE_STEP) / (BASELINE_STEPS - TARGET_STEPS)


def _write_curve_trial(root: Path, curve: dict[str, list], *, reward_full=None) -> Path:
    """A trial dir carrying only seed logs, in the format `parent_artifacts` parses."""
    logs = root / "artifacts" / "workspace" / "submission" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for seed, points in curve.items():
        logs.joinpath(f"full_seed{seed}.log").write_text(
            "".join(f"step:{s}/3500 val_loss:{v:.5f} train_time:0.0s\n"
                    for s, v in points)
        )
    if reward_full is not None:
        (root / "verifier").mkdir(parents=True, exist_ok=True)
        (root / "verifier" / "reward_full.json").write_text(json.dumps(reward_full))
    return root

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


def test_read_trial_reads_reason_from_reward_full_not_reward_json():
    """reward.json is a sibling judge module's input; the parser reads reward_full.

    Still true after the unwiring: reward_full is the only file carrying `reason`,
    and keeping it is what makes a verifier-graded trial diagnosable. Only the
    *reward* stopped coming from there.
    """
    row = read_trial(FIXTURE)
    assert row["reason"] == "graded_step=3200"


def test_read_trial_reward_comes_from_the_curve_not_the_verifier():
    """The fixture's verifier says 0.5; the logs say 3175 steps, so 325/600."""
    row = read_trial(FIXTURE)
    verifier = json.loads((FIXTURE / "verifier" / "reward_full.json").read_text())
    assert verifier["reward"] == 0.5, "fixture drifted; the point is that we ignore it"
    assert row["reward"] == pytest.approx(FIXTURE_REWARD)
    assert row["reward"] == pytest.approx(0.5416666666)
    assert row["reward"] != 0.5


def test_read_trial_graded_step_comes_from_the_full_density_logs():
    """3175, the slowest seed's first crossing in the RAW logs — not 3250, not 3200."""
    row = read_trial(FIXTURE)
    assert row["graded_step"] == FIXTURE_STEP == 3175
    assert row["reward"] == pytest.approx(compute_reward(row["graded_step"]))


def test_read_trial_reward_agrees_with_reward_from_curve_on_the_full_curve():
    """The row's reward must be exactly what the module computes from `full_curve`."""
    row = read_trial(FIXTURE)
    expected_reward, expected_step = reward_from_curve(
        parent_artifacts(FIXTURE)["full_curve"])
    assert row["reward"] == pytest.approx(expected_reward)
    assert row["graded_step"] == expected_step


def test_the_real_fixture_is_itself_a_case_thinning_gets_wrong():
    """The production fixture is not a synthetic edge case — it is the bug, verbatim.

    Both seeds cross 3.28 at a step the display stride drops, so the thinned curve
    under-reports the run by a full 0.125 of reward. If this assertion ever starts
    failing because the two agree, the regression tests above are the ones still
    guarding the property; this one just records that real runs hit it routinely.
    """
    artifacts = parent_artifacts(FIXTURE)
    full_reward, full_step = reward_from_curve(artifacts["full_curve"])
    thin_reward, thin_step = reward_from_curve(artifacts["parent_curve"])

    assert full_step == FIXTURE_STEP == 3175
    assert thin_step == FIXTURE_THINNED_STEP == 3250
    assert full_reward == pytest.approx(0.5416666666)
    assert thin_reward == pytest.approx(0.4166666666)
    assert read_trial(FIXTURE)["reward"] == pytest.approx(full_reward)


def test_per_seed_first_crossings_match_the_raw_logs():
    """Re-derived straight from the log text, bypassing `parent_artifacts` entirely."""
    import re

    crossings = {}
    for log in FIXTURE.glob("artifacts/**/full_seed*.log"):
        seed = log.stem.replace("full_seed", "")
        pts = [(int(s), float(v)) for s, v in re.findall(
            r"step:(\d+)/\d+ val_loss:([\d.]+)", log.read_text(errors="replace"))]
        assert len(pts) == 61, f"seed {seed} has {len(pts)} points"
        crossings[seed] = next(s for s, v in pts if v <= 3.28)

    assert crossings == {"0": 3150, "1": 3175}
    assert read_trial(FIXTURE)["graded_step"] == max(crossings.values())


def test_read_trial_classifies_graded_pass_without_the_verifier():
    """`classify` receives the COMPUTED reward, so a real run passes on its own."""
    row = read_trial(FIXTURE)
    assert row["outcome"] == "graded_pass"


def test_read_trial_on_a_synthetic_run_that_crosses_the_target(tmp_path):
    """A run whose logs cross 3.28 at 3200 on the slowest seed scores exactly 0.5."""
    trial = _write_curve_trial(tmp_path / "t", {
        "0": [(3000, 3.31), (3100, 3.27), (3200, 3.26)],
        "1": [(3000, 3.32), (3100, 3.29), (3200, 3.27)],
    })
    row = read_trial(trial)
    assert row["graded_step"] == 3200
    assert row["reward"] == pytest.approx(0.5)
    assert row["outcome"] == "graded_pass"


def test_reward_is_measured_on_full_density_points_not_the_thinned_curve(tmp_path):
    """REGRESSION: `MAX_CURVE_POINTS` is a DISPLAY constant. It must not move the score.

    `parent_curve` is thinned only so the history markdown stays small enough to inject
    into the agent's prompt. If the reward were read off that thinned curve, the stride
    would be load-bearing: thinning can only ever miss the true first crossing or find
    it late, never early, so every retuning of the display constant would silently
    re-score the whole campaign — downwards.

    This trial is built so that both seeds cross the target at a step the stride drops.
    61 points -> stride 61 // 12 = 5 -> only indices 0, 5, 10, ... survive, so a
    crossing at index 7 or 12 is invisible to the thinned curve, which sees the next
    retained point instead. The test pins the FULL-density answer and separately shows
    that the thinned curve still gives the wrong one, so it cannot pass by accident.
    """
    steps = [2900 + 10 * i for i in range(61)]
    seed_cross_idx = {"0": 7, "1": 12}  # neither is a multiple of the stride
    trial = _write_curve_trial(tmp_path / "t", {
        seed: [(s, 3.30 if i < idx else 3.27) for i, s in enumerate(steps)]
        for seed, idx in seed_cross_idx.items()
    })

    true_step = max(steps[i] for i in seed_cross_idx.values())
    assert true_step == 3020
    row = read_trial(trial)

    assert row["graded_step"] == true_step
    assert row["reward"] == pytest.approx((BASELINE_STEPS - true_step) /
                                          (BASELINE_STEPS - TARGET_STEPS))
    assert row["reward"] == pytest.approx(0.8)

    # And the thinned display curve genuinely disagrees -- i.e. this test is exercising
    # the property, not restating an equality that holds either way.
    thinned_reward, thinned_step = reward_from_curve(row["parent_curve"])
    assert thinned_step == 3050, "fixture drifted; the stride must skip the crossing"
    assert thinned_step > true_step
    assert thinned_reward < row["reward"]


def test_thinning_stride_cannot_change_the_reward_at_any_density(tmp_path):
    """The same crossing, logged at three densities, must score the same every time.

    Density changes which points thinning keeps; it must not change the score. The
    coarsest run here is short enough that no thinning happens at all, which pins the
    full-density reading to the one case where the two curves cannot differ.
    """
    rewards, thinned_steps = set(), set()
    for stride in (100, 20, 4, 2):  # each divides 300, so 3200 is on every grid
        steps = list(range(2900, 3501, stride))
        assert 3200 in steps
        trial = _write_curve_trial(tmp_path / f"t{stride}", {
            seed: [(s, 3.27 if s >= 3200 else 3.30) for s in steps]
            for seed in ("0", "1")
        })
        row = read_trial(trial)
        assert row["graded_step"] == 3200, stride
        rewards.add(round(row["reward"], 12))
        thinned_steps.add(reward_from_curve(row["parent_curve"])[1])

    assert rewards == {0.5}, f"density changed the reward: {rewards}"
    assert thinned_steps != {3200}, "no density here actually stresses the thinning"


@pytest.mark.parametrize("max_points", [1, 2, 3, 5, 12, 20, 61, 500])
def test_max_curve_points_is_not_load_bearing_on_the_reward(monkeypatch, max_points):
    """Retune the display budget however you like; the REAL fixture's score is fixed.

    This is the direct statement of the invariant, on the production fixture rather
    than a constructed one: `MAX_CURVE_POINTS` governs how much curve fits in the
    agent's prompt and NOTHING else. At 1 the thinned curve collapses to two points
    and would score 0.0 if it were the input; the reward must not move.
    """
    import track3.trial_io as trial_io

    monkeypatch.setattr(trial_io, "MAX_CURVE_POINTS", max_points)
    row = read_trial(FIXTURE)

    assert row["graded_step"] == FIXTURE_STEP
    assert row["reward"] == pytest.approx(FIXTURE_REWARD)

    # The knob demonstrably moved the display curve -- and the score still did not.
    artifacts = parent_artifacts(FIXTURE)
    assert all(len(pts) == 61 for pts in artifacts["full_curve"].values())
    kept = {len(pts) for pts in artifacts["parent_curve"].values()}
    if max_points == 1:
        assert kept == {2}, "the stride did not collapse the curve as expected"
        assert reward_from_curve(artifacts["parent_curve"])[0] == 0.0, \
            "scoring the thinned curve here would have wiped the run out entirely"
    elif max_points >= 61:
        assert kept == {61}


def test_read_trial_on_a_synthetic_run_reaching_target_steps(tmp_path):
    trial = _write_curve_trial(tmp_path / "t", {
        "0": [(2900, 3.27), (3000, 3.26)],
        "1": [(2900, 3.26), (3000, 3.25)],
    })
    row = read_trial(trial)
    assert row["graded_step"] == 2900
    assert row["reward"] == 1.0


def test_read_trial_scores_zero_when_a_single_seed_misses(tmp_path):
    """AGREEMENT: one lucky seed may not carry a failing one."""
    trial = _write_curve_trial(tmp_path / "t", {
        "0": [(3000, 3.27), (3500, 3.26)],
        "1": [(3000, 3.35), (3500, 3.31)],
    })
    row = read_trial(trial)
    assert row["graded_step"] is None
    assert row["reward"] == 0.0
    assert row["outcome"] != "graded_pass"


def test_read_trial_ignores_a_verifier_reward_that_disagrees_with_the_curve(tmp_path):
    """A stale or generous reward_full cannot inflate the row any more.

    The reward is the formula's and only the formula's. `graded_step` is the one
    field that still defers to the verifier, and only when a curve exists but yields
    no crossing of its own — a step the thinning dropped is worth reporting even
    though it earns nothing.
    """
    trial = _write_curve_trial(
        tmp_path / "t",
        {"0": [(3400, 3.40)], "1": [(3400, 3.41)]},
        reward_full={"reward": 1.0, "reason": "graded_step=2900"},
    )
    row = read_trial(trial)
    assert row["reward"] == 0.0
    assert row["reason"] == "graded_step=2900"
    assert row["graded_step"] == 2900


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


def test_full_curve_is_consumed_for_scoring_and_dropped_from_the_row():
    """It scores the row, then it goes. A 61-point-per-seed array is ledger bloat.

    `ledger.jsonl` is append-only and read every iteration by `render_history`, which
    only ever touches `parent_curve`. Persisting `full_curve` would multiply each row's
    curve payload by ~5 for no reader, over a campaign of hundreds of iterations. The
    reward it produced is the durable artifact; the points are re-derivable from the
    trial's own logs, whose path the row already records in `trial_dir`.
    """
    row = read_trial(FIXTURE)
    assert "full_curve" not in row
    assert "parent_curve" in row
    assert row["trial_dir"] == str(FIXTURE)
    assert EXPECTED_KEYS <= set(row)


def test_ledger_row_stays_json_serialisable_and_carries_only_the_thinned_curve():
    row = read_trial(FIXTURE)
    reloaded = json.loads(json.dumps(row))
    assert "full_curve" not in reloaded
    for seed, pts in reloaded["parent_curve"].items():
        assert len(pts) <= 13, f"seed {seed} would bloat the ledger with {len(pts)}"


def test_read_trial_on_empty_dir_does_not_raise(tmp_path):
    row = read_trial(tmp_path)
    assert row["reward"] == 0.0
    assert row["reason"] == "no_curve"
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
    assert row["graded_step"] is None


def test_read_trial_keeps_the_verifier_reason_when_no_curve_exists(tmp_path):
    """No curve is not silence: the row must still say why it scored nothing."""
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier" / "reward_full.json").write_text(
        json.dumps({"reward": 0.0, "reason": "telemetry_absent"})
    )
    row = read_trial(tmp_path)
    assert row["reason"] == "telemetry_absent"
    assert row["reward"] == 0.0
    assert row["graded_step"] is None


def test_read_trial_still_reads_verifier_metrics_when_present(tmp_path):
    """reward_full stays wired for `reason` and `metrics`; only its reward is dropped."""
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier" / "reward_full.json").write_text(
        json.dumps({"reward": 0.9, "reason": "graded_step=3000",
                    "metrics": {"n_seeds": 4}})
    )
    row = read_trial(tmp_path)
    assert row["n_seeds"] == 4
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
    """The prompt budget. `parent_curve` is what gets rendered into the agent's
    history markdown, so it must stay small even though the reward no longer uses it."""
    curve = parent_artifacts(FIXTURE)["parent_curve"]
    for seed, pts in curve.items():
        assert len(pts) <= 13, f"seed {seed} kept {len(pts)} points"
        assert pts, f"seed {seed} is empty"


def test_full_curve_keeps_every_parsed_point_and_parent_curve_is_a_subset():
    artifacts = parent_artifacts(FIXTURE)
    full, thinned = artifacts["full_curve"], artifacts["parent_curve"]
    assert set(full) == set(thinned) == {"0", "1"}
    for seed, pts in full.items():
        assert len(pts) == 61, f"seed {seed} kept {len(pts)} of 61 points"
        assert pts == sorted(pts), f"seed {seed} is out of order"
        assert set(map(tuple, thinned[seed])) <= set(map(tuple, pts))
        assert len(thinned[seed]) < len(pts), "thinning is meant to drop points"


def test_full_curve_is_absent_exactly_when_parent_curve_is(tmp_path):
    assert "full_curve" not in parent_artifacts(tmp_path)
    log = tmp_path / "artifacts" / "logs" / "full_seed0.log"
    log.parent.mkdir(parents=True)
    log.write_text("nothing parseable here\n")
    assert "full_curve" not in parent_artifacts(tmp_path)


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


def test_find_trial_honours_the_five_second_grace(tmp_path):
    now = time.time()
    job = _job_with_trials(tmp_path / "job", [("a", now - 3)])
    assert find_trial(job, now) == job / "a"


# --------------------------------------------------------------------------
# DEFECT 1: a stale trial must never be scored as this iteration's result
# --------------------------------------------------------------------------


def _stale_job(tmp_path):
    """A job dir whose every trial predates the launch, and that launch time.

    Staleness is expressed by launching AFTER everything on disk rather than by
    backdating the trials, because `os.utime` cannot backdate ctime -- the kernel
    bumps it on any metadata write -- and `find_trial` deliberately treats a fresh
    ctime as proof of arrival. A launch time later than every candidate is the same
    situation the defect describes (relaunch of an `agentic_iterNN` dir that already
    holds an earlier run's trial) and is expressible with stdlib alone.
    """
    now = time.time()
    job = _job_with_trials(
        tmp_path / "job", [("older", now - 9000), ("newer", now - 6000)]
    )
    return job, now + 3600


def test_find_trial_does_not_return_a_stale_trial_by_default(tmp_path):
    """The whole defect. A failed relaunch produced NO trial this iteration.

    Returning an earlier run's trial from the same agentic_iterNN job dir makes the
    loop score a plausible success where there was a launch failure -- reward, curve
    and carried parent source all belong to a different iteration.
    """
    job, since = _stale_job(tmp_path)
    assert find_trial(job, since) is None


def test_find_trial_warns_loudly_on_stderr_when_the_stale_path_is_taken(tmp_path,
                                                                        capsys):
    job, since = _stale_job(tmp_path)
    find_trial(job, since)
    err = capsys.readouterr().err
    assert "stale" in err.lower()
    assert "newer" in err


def test_find_trial_does_not_warn_when_a_fresh_trial_exists(tmp_path, capsys):
    now = time.time()
    job = _job_with_trials(tmp_path / "job", [("old", now - 9000), ("new", now)])
    assert find_trial(job, now) == job / "new"
    assert capsys.readouterr().err == ""


def test_find_trial_does_not_warn_for_an_empty_job_dir(tmp_path, capsys):
    """Nothing at all is honestly None already -- no stale path was taken."""
    job = tmp_path / "job"
    job.mkdir()
    assert find_trial(job, time.time()) is None
    assert capsys.readouterr().err == ""


def test_find_trial_returns_the_stale_trial_only_on_explicit_opt_in(tmp_path):
    """The fallback is kept -- a stale trial is still diagnostically useful."""
    job, since = _stale_job(tmp_path)
    assert find_trial(job, since, allow_stale=True) == job / "newer"


def test_find_trial_with_staleness_flags_the_stale_case(tmp_path):
    job, since = _stale_job(tmp_path)
    assert find_trial_with_staleness(job, since) == (job / "newer", True)


def test_find_trial_with_staleness_flags_the_fresh_case(tmp_path):
    now = time.time()
    job = _job_with_trials(tmp_path / "job", [("old", now - 9000), ("new", now)])
    assert find_trial_with_staleness(job, now) == (job / "new", False)


def test_a_copied_in_trial_is_fresh_despite_a_backdated_mtime(tmp_path):
    """`shutil.copytree` preserves the SOURCE mtime; ctime tells the truth.

    This is how harbor trials arrive in the loop's own test doubles, and it is how
    any archive/rsync-delivered trial arrives in production. Judging arrival on mtime
    alone would report a directory created milliseconds ago as an earlier run's.
    """
    import shutil
    src = tmp_path / "src"
    src.mkdir()
    (src / "result.json").write_text("{}")
    os.utime(src, (time.time() - 90000, time.time() - 90000))

    job = tmp_path / "job"
    job.mkdir()
    launched_at = time.time()
    shutil.copytree(src, job / "trial-1")

    assert (job / "trial-1").stat().st_mtime < launched_at - 80000
    assert find_trial_with_staleness(job, launched_at) == (job / "trial-1", False)


def test_ordering_still_uses_mtime_not_arrival(tmp_path):
    """Arrival decides fresh-vs-stale; mtime alone ranks trials against each other."""
    now = time.time()
    job = _job_with_trials(tmp_path / "job", [("first", now - 100), ("second", now)])
    assert find_trial(job, now - 500) == job / "second"


def test_find_trial_with_staleness_on_empty_dir_is_not_stale(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    assert find_trial_with_staleness(job, time.time()) == (None, False)


def test_find_trial_default_call_signature_still_takes_two_positionals(tmp_path):
    """loop.py calls find_trial(job_dir, t0) and is owned by another agent."""
    import inspect
    params = list(inspect.signature(find_trial).parameters.values())
    positional = [p for p in params
                  if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert [p.name for p in positional[:2]] == ["job_dir", "since"]
    assert all(p.default is inspect.Parameter.empty for p in positional[:2])
    now = time.time()
    job = _job_with_trials(tmp_path / "job", [("new", now)])
    assert find_trial(job, now - 200) == job / "new"


# --------------------------------------------------------------------------
# DEFECT 1: a directory vanishing between the glob and the stat
# --------------------------------------------------------------------------


def test_find_trial_survives_a_directory_vanishing_between_glob_and_stat(tmp_path):
    """A concurrent reaper can unlink a trial after glob() yielded it."""
    now = time.time()
    job = _job_with_trials(tmp_path / "job", [("doomed", now), ("survivor", now)])
    doomed = job / "doomed"
    real_stat = Path.stat

    def flaky_stat(self, *a, **kw):
        if self == doomed:
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_stat(self, *a, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "stat", flaky_stat)
        assert find_trial(job, now - 200) == job / "survivor"


def test_find_trial_returns_none_when_every_directory_vanishes(tmp_path):
    now = time.time()
    job = _job_with_trials(tmp_path / "job", [("a", now), ("b", now)])

    doomed = {job / "a", job / "b"}
    real_stat = Path.stat

    def gone_stat(self, *a, **kw):
        if self in doomed:
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_stat(self, *a, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "stat", gone_stat)
        assert find_trial(job, now - 200) is None


def test_find_trial_survives_a_vanish_on_the_stale_path_too(tmp_path):
    job, since = _stale_job(tmp_path)
    doomed = job / "older"
    real_stat = Path.stat

    def flaky_stat(self, *a, **kw):
        if self == doomed:
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_stat(self, *a, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "stat", flaky_stat)
        assert find_trial(job, since, allow_stale=True) == job / "newer"


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
