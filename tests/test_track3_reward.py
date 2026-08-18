"""The step-derived reward, computed from the loss curve rather than the verifier.

The verifier and the LLM judge are currently UNWIRED (see `runner/track3/loop.py`),
so `track3.reward` is the only thing that turns a run into a number. That makes two
properties load-bearing and worth asserting directly:

* THE FORMULA IS LINEAR AND CLAMPED. `(BASELINE_STEPS - step) / (BASELINE_STEPS -
  TARGET_STEPS)` with a 600-step denominator: 3500 -> 0.0, 3200 -> 0.5, 2900 -> 1.0.
  Beating the target must not pay more than 1.0 and missing the baseline must not pay
  less than 0.0, or a single wild run dominates every comparison in the ledger.
* AGREEMENT ACROSS SEEDS. `tests/grade.py::graded_step` documents "every seed must
  individually reach TARGET_LOSS", because a mean-only rule lets one lucky seed carry
  a failing one. So the crossing step is the MAX over seeds, not the min and not the
  mean, and one seed that never crosses makes the whole run ungraded (None).
"""

from pathlib import Path

import pytest

from track3.reward import (
    BASELINE_STEPS,
    MIRRORED_FROM_GRADE_PY,
    TARGET_LOSS,
    TARGET_STEPS,
    compute_reward,
    constants_from_grade_py,
    reward_from_curve,
    step_to_target,
)


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #


def test_constants_match_the_task_grader():
    """These are mirrored from tasks/<uuid>/tests/grade.py; drift is a silent bug."""
    assert BASELINE_STEPS == 3500
    assert TARGET_STEPS == 2900
    assert TARGET_LOSS == 3.28
    assert BASELINE_STEPS - TARGET_STEPS == 600


# --------------------------------------------------------------------------- #
# DRIFT DETECTION -- reward.py mirrors the task grader; nothing enforced that
# --------------------------------------------------------------------------- #
#
# `reward.py` copies BASELINE_STEPS/TARGET_STEPS/TARGET_LOSS out of
# `tasks/<uuid>/tests/grade.py` by hand. If the task bundle's numbers are ever
# edited, the mirror keeps scoring on the OLD ones: every reward in the campaign
# is wrong and NOTHING raises. The tests below are the only error surface that
# defect has, so they must fail loudly and name both values.
#
# grade.py is PARSED, never imported: it is stdlib-only in-container code that
# reads env vars and writes reward.json at import-adjacent scope, and the outer
# loop must not execute the task's test tree to score a run.

REPO_ROOT = Path(__file__).resolve().parents[1]


def _grade_py_task_dirs():
    """Task bundles on disk that carry the grader these constants come from."""
    return sorted(
        p.parent.parent for p in REPO_ROOT.glob("tasks/*/tests/grade.py")
        if "BASELINE_STEPS" in p.read_text(encoding="utf-8", errors="replace")
    )


def _assert_no_drift(parsed: dict, source: str) -> None:
    """The comparison itself, factored out so the detector can be tested too."""
    mirrored = {"BASELINE_STEPS": BASELINE_STEPS,
                "TARGET_STEPS": TARGET_STEPS,
                "TARGET_LOSS": TARGET_LOSS}
    drifted = [
        f"  {name}: runner/track3/reward.py has {mirrored[name]!r}, "
        f"{source} has {parsed[name]!r}"
        for name in MIRRORED_FROM_GRADE_PY if mirrored[name] != parsed[name]
    ]
    assert not drifted, (
        "track3.reward constants have DRIFTED from the task grader that owns them:\n"
        + "\n".join(drifted)
        + "\nEvery reward in the campaign is being scored on stale numbers, with no "
          "error surface. Update runner/track3/reward.py to match the task bundle "
          "(or fix the bundle), then re-run."
    )


def test_reward_constants_do_not_drift_from_the_task_grader():
    """The mirror agrees with the source of truth TODAY. Fails the day it stops."""
    task_dirs = _grade_py_task_dirs()
    if not task_dirs:
        pytest.skip(
            "no task bundle with tests/grade.py defining BASELINE_STEPS is present "
            "under tasks/; the mirror cannot be checked without it, and the repo is "
            "meant to stay testable with the bundle absent"
        )
    for task_dir in task_dirs:
        _assert_no_drift(
            constants_from_grade_py(task_dir),
            f"{task_dir.relative_to(REPO_ROOT)}/tests/grade.py",
        )


def test_drift_check_fails_loudly_when_the_grader_moves(tmp_path):
    """Prove the detector detects.

    The real bundle is read-only, so the divergence is staged in a temp copy of
    grade.py rather than by editing tasks/. Asserts on the FAILURE MESSAGE: a
    drift test that fails without naming both numbers sends whoever hits it back
    to the source anyway.
    """
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "grade.py").write_text(
        "import os\n"
        "BASELINE_STEPS = 3400\n"
        "TARGET_STEPS = 2900\n"
        "TARGET_LOSS = 3.20\n"
        "SIG_MARGIN = 0.004\n"
    )
    parsed = constants_from_grade_py(tmp_path)
    with pytest.raises(AssertionError) as exc:
        _assert_no_drift(parsed, "tampered/grade.py")
    msg = str(exc.value)
    assert "DRIFTED" in msg
    # both sides of every divergence are named, and the agreeing one is not noise
    assert "3400" in msg and str(BASELINE_STEPS) in msg
    assert "3.2" in msg and str(TARGET_LOSS) in msg
    assert "TARGET_STEPS" not in msg


# --------------------------------------------------------------------------- #
# constants_from_grade_py -- the parser behind the drift check
# --------------------------------------------------------------------------- #


def test_constants_from_grade_py_reads_the_real_bundle():
    task_dirs = _grade_py_task_dirs()
    if not task_dirs:
        pytest.skip("no task bundle with tests/grade.py on disk")
    parsed = constants_from_grade_py(task_dirs[0])
    assert set(parsed) == set(MIRRORED_FROM_GRADE_PY)
    assert isinstance(parsed["BASELINE_STEPS"], int)
    assert isinstance(parsed["TARGET_LOSS"], float)


def test_constants_from_grade_py_accepts_the_tests_dir_or_the_file(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    grade = tests_dir / "grade.py"
    grade.write_text("BASELINE_STEPS = 10\nTARGET_STEPS = 5\nTARGET_LOSS = 1.5\n")
    expected = {"BASELINE_STEPS": 10, "TARGET_STEPS": 5, "TARGET_LOSS": 1.5}
    assert constants_from_grade_py(tmp_path) == expected
    assert constants_from_grade_py(tests_dir) == expected
    assert constants_from_grade_py(grade) == expected


def test_constants_from_grade_py_does_not_execute_the_grader(tmp_path):
    """Parsing, not importing: in-container grader code must never run out here."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "grade.py").write_text(
        "raise SystemExit('grade.py must never be executed by the outer loop')\n"
        "BASELINE_STEPS = 3500\nTARGET_STEPS = 2900\nTARGET_LOSS = 3.28\n"
    )
    assert constants_from_grade_py(tmp_path)["BASELINE_STEPS"] == 3500


def test_constants_from_grade_py_ignores_nested_rebindings(tmp_path):
    """Only module-level assignments are the contract; a local shadow is not."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "grade.py").write_text(
        "BASELINE_STEPS = 3500\nTARGET_STEPS = 2900\nTARGET_LOSS = 3.28\n"
        "def f():\n    BASELINE_STEPS = 1\n    return BASELINE_STEPS\n"
    )
    assert constants_from_grade_py(tmp_path)["BASELINE_STEPS"] == 3500


def test_constants_from_grade_py_raises_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        constants_from_grade_py(tmp_path)


def test_constants_from_grade_py_raises_when_a_constant_is_missing(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "grade.py").write_text("BASELINE_STEPS = 3500\n")
    with pytest.raises(ValueError) as exc:
        constants_from_grade_py(tmp_path)
    assert "TARGET_STEPS" in str(exc.value)


# --------------------------------------------------------------------------- #
# compute_reward
# --------------------------------------------------------------------------- #


def test_compute_reward_at_baseline_is_zero():
    assert compute_reward(3500) == 0.0


def test_compute_reward_at_target_is_one():
    assert compute_reward(2900) == 1.0


def test_compute_reward_at_midpoint_is_one_half():
    assert compute_reward(3200) == pytest.approx(0.5)


def test_compute_reward_clamps_above_target():
    """Beating the target cannot pay more than full marks."""
    assert compute_reward(2500) == 1.0


def test_compute_reward_clamps_below_baseline():
    """A run slower than the baseline scores 0.0, never a negative number."""
    assert compute_reward(4000) == 0.0


def test_compute_reward_of_none_is_zero():
    """No crossing is not a small reward; it is no reward."""
    assert compute_reward(None) == 0.0


def test_compute_reward_accepts_floats():
    assert compute_reward(3200.0) == pytest.approx(0.5)


def test_compute_reward_is_linear_between_the_anchors():
    for step, expected in ((3400, 100 / 600), (3100, 400 / 600), (2950, 550 / 600)):
        assert compute_reward(step) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# step_to_target -- AGREEMENT across seeds
# --------------------------------------------------------------------------- #


def test_step_to_target_first_step_at_or_below_target():
    curve = {"0": [[3000, 3.30], [3200, 3.27]],
             "1": [[3000, 3.29], [3200, 3.26]]}
    assert step_to_target(curve) == 3200


def test_step_to_target_takes_the_max_across_seeds():
    """The run has only reached target once the SLOWEST seed has."""
    curve = {"0": [[3100, 3.27], [3300, 3.26]],
             "1": [[3100, 3.29], [3300, 3.25]]}
    assert step_to_target(curve) == 3300


def test_step_to_target_is_none_when_any_seed_never_crosses():
    """One lucky seed may not carry a failing one -- grade.py's AGREEMENT rule."""
    curve = {"0": [[3100, 3.27], [3300, 3.26]],
             "1": [[3100, 3.31], [3300, 3.30]]}
    assert step_to_target(curve) is None


def test_step_to_target_is_none_for_empty_or_missing_curve():
    assert step_to_target({}) is None
    assert step_to_target(None) is None


def test_step_to_target_ignores_a_later_relapse():
    """FIRST crossing per seed, so a dip that is later given back still counts here."""
    curve = {"0": [[3000, 3.27], [3200, 3.40]],
             "1": [[3000, 3.26], [3200, 3.41]]}
    assert step_to_target(curve) == 3000


def test_step_to_target_boundary_is_inclusive():
    curve = {"0": [[3200, TARGET_LOSS]], "1": [[3200, TARGET_LOSS]]}
    assert step_to_target(curve) == 3200


def test_step_to_target_honours_a_custom_target_loss():
    curve = {"0": [[3000, 3.30], [3200, 3.27]],
             "1": [[3000, 3.29], [3200, 3.26]]}
    assert step_to_target(curve, target_loss=3.30) == 3000
    assert step_to_target(curve, target_loss=3.00) is None


def test_step_to_target_tolerates_a_seed_with_no_points():
    curve = {"0": [[3200, 3.27]], "1": []}
    assert step_to_target(curve) is None


def test_step_to_target_accepts_tuples_as_well_as_lists():
    """parent_curve is tuples in memory and lists after a JSON round-trip."""
    curve = {"0": [(3200, 3.27)], "1": [(3200, 3.26)]}
    assert step_to_target(curve) == 3200


def test_step_to_target_does_not_assume_sorted_input():
    curve = {"0": [[3300, 3.26], [3100, 3.27]],
             "1": [[3300, 3.25], [3100, 3.29]]}
    assert step_to_target(curve) == 3300


def test_step_to_target_survives_malformed_points():
    """A partial trial is normal input; a parser that raises turns it into a dead loop."""
    curve = {"0": [["junk", 3.27], [3200, 3.27]], "1": [[3200, 3.26]]}
    assert step_to_target(curve) == 3200


# --------------------------------------------------------------------------- #
# reward_from_curve
# --------------------------------------------------------------------------- #


def test_reward_from_curve_returns_the_matching_pair():
    curve = {"0": [[3000, 3.30], [3200, 3.27]],
             "1": [[3000, 3.29], [3200, 3.26]]}
    reward, step = reward_from_curve(curve)
    assert step == 3200
    assert reward == pytest.approx(0.5)
    assert reward == compute_reward(step_to_target(curve))


def test_reward_from_curve_on_a_run_that_never_crosses():
    curve = {"0": [[3500, 3.40]], "1": [[3500, 3.41]]}
    assert reward_from_curve(curve) == (0.0, None)


def test_reward_from_curve_on_an_empty_curve():
    assert reward_from_curve({}) == (0.0, None)
    assert reward_from_curve(None) == (0.0, None)


def test_reward_from_curve_full_marks_at_target_steps():
    curve = {"0": [[2900, 3.27]], "1": [[2900, 3.26]]}
    assert reward_from_curve(curve) == (1.0, 2900)
