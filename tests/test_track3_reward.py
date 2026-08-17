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

import pytest

from track3.reward import (
    BASELINE_STEPS,
    TARGET_LOSS,
    TARGET_STEPS,
    compute_reward,
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
