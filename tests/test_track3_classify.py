"""Tests for track3.classify, ported from track3-pipeline/tools/refine.py.

Branch ORDER is load-bearing: the classification decides what the next
iteration is told about the previous one, so each branch is pinned
independently rather than only through happy-path cases.
"""

import pytest

from track3.classify import classify, _graded_step


# --- the six outcomes ---------------------------------------------------


def test_graded_pass():
    assert classify(0.58, "", 2) == "graded_pass"


def test_agent_abandoned_run():
    assert classify(0.0, "telemetry_absent", 1, harness_error=False,
                    budget_used_frac=0.3) == "agent_abandoned_run"


def test_harness_incomplete():
    assert classify(0.0, "telemetry_absent", 1, harness_error=True,
                    budget_used_frac=0.3) == "harness_incomplete"


def test_graded_miss():
    assert classify(0.0, "no_step_clears", 2) == "graded_miss"


def test_gate_fail():
    assert classify(0.0, "chain_mismatch", 2) == "gate_fail"


def test_unknown():
    assert classify(0.0, "something weird", 2) == "unknown"


# --- traps: what flips agent_abandoned_run vs harness_incomplete --------


def test_harness_error_flips_abandoned_to_incomplete():
    """Same inputs as the abandoned case except harness_error=True."""
    abandoned = classify(0.0, "telemetry_absent", 1, harness_error=False,
                         budget_used_frac=0.3)
    flipped = classify(0.0, "telemetry_absent", 1, harness_error=True,
                       budget_used_frac=0.3)
    assert abandoned == "agent_abandoned_run"
    assert flipped == "harness_incomplete"


def test_budget_spent_flips_abandoned_to_incomplete():
    """An agent that spent its budget did not abandon the run."""
    assert classify(0.0, "telemetry_absent", 1, harness_error=False,
                    budget_used_frac=0.9) == "harness_incomplete"


def test_budget_boundary_is_exclusive_at_0_8():
    # unspent is strictly < 0.8
    assert classify(0.0, "telemetry_absent", 1, harness_error=False,
                    budget_used_frac=0.79) == "agent_abandoned_run"
    assert classify(0.0, "telemetry_absent", 1, harness_error=False,
                    budget_used_frac=0.8) == "harness_incomplete"


def test_unknown_budget_is_not_abandonment():
    """budget_used_frac=None means we cannot claim the agent quit early."""
    assert classify(0.0, "telemetry_absent", 1,
                    harness_error=False) == "harness_incomplete"


# --- traps: the ungradable predicate ------------------------------------


@pytest.mark.parametrize("reason", ["telemetry_absent", "seed_mismatch",
                                    "not_bound", "SEED", "TELEMETRY_ABSENT"])
def test_ungradable_reasons_are_not_graded(reason):
    """These reasons mean 'no verdict', never graded_miss/gate_fail."""
    assert classify(0.0, reason, 2, harness_error=False,
                    budget_used_frac=0.9) == "harness_incomplete"


def test_too_few_seeds_is_ungradable_regardless_of_reason():
    """n_seeds < 2 outranks a reason that would otherwise be graded_miss."""
    assert classify(0.0, "no_step_clears", 1, harness_error=False,
                    budget_used_frac=0.9) == "harness_incomplete"


def test_ungradable_outranks_gate_fail():
    """'seed' in reason wins over the 'chain' gate_fail branch."""
    assert classify(0.0, "chain_mismatch_seed", 2, harness_error=False,
                    budget_used_frac=0.9) == "harness_incomplete"


# --- traps: the `reward and reward > 0` guard ---------------------------


def test_reward_exactly_zero_does_not_short_circuit():
    """0.0 is falsy: it must fall through to the reason-based branches."""
    assert classify(0.0, "", 2) != "graded_pass"
    assert classify(0.0, "", 2) == "unknown"
    assert classify(0.0, "no_step_clears", 2) == "graded_miss"


def test_reward_none_does_not_short_circuit():
    assert classify(None, "no_step_clears", 2) == "graded_miss"


def test_negative_reward_is_not_a_pass():
    assert classify(-1.0, "no_step_clears", 2) == "graded_miss"


def test_tiny_positive_reward_is_a_pass():
    assert classify(0.0001, "no_step_clears", 2) == "graded_pass"


def test_graded_pass_outranks_ungradable():
    """A positive reward wins even with 1 seed."""
    assert classify(0.58, "telemetry_absent", 1) == "graded_pass"


# --- traps: reason normalisation ----------------------------------------


def test_reason_none_is_tolerated():
    assert classify(0.0, None, 2) == "unknown"


@pytest.mark.parametrize("reason", ["frozen", "reconciliation_error",
                                    "chain_mismatch", "FROZEN", "Reconcil"])
def test_gate_fail_reasons(reason):
    assert classify(0.0, reason, 2) == "gate_fail"


def test_no_step_clears_is_case_insensitive():
    assert classify(0.0, "NO_STEP_CLEARS", 2) == "graded_miss"


def test_graded_miss_outranks_gate_fail():
    """no_step_clears is checked before the gate_fail reasons."""
    assert classify(0.0, "no_step_clears_chain", 2) == "graded_miss"


# --- _graded_step -------------------------------------------------------


def test_graded_step_parses_int():
    assert _graded_step("graded_step=3275") == 3275


def test_graded_step_non_numeric_is_none():
    assert _graded_step("graded_step=abc") is None


def test_graded_step_unrelated_reason_is_none():
    assert _graded_step("other") is None


def test_graded_step_empty_and_none():
    assert _graded_step("") is None
    assert _graded_step(None) is None


def test_graded_step_splits_on_first_equals_only():
    assert _graded_step("graded_step=12=34") is None
