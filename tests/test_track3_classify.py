"""Tests for track3.classify, ported from track3-pipeline/tools/refine.py.

Branch ORDER is load-bearing: the classification decides what the next
iteration is told about the previous one, so each branch is pinned
independently rather than only through happy-path cases.
"""

import inspect
import re

import pytest

from track3 import classify as classify_mod
from track3.classify import RECOGNISED_REASON_TOKENS, classify, _graded_step


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


def test_unknown_budget_must_not_be_blamed_on_the_harness_when_harbor_logged_no_error():
    """DEFECT 2. budget_used_frac=None means "we could not measure", NOT "spent".

    `harness_incomplete` is not a neutral don't-know bucket: it is an affirmative
    claim of harness fault, and it is the bucket `history.py` uses to tell the next
    iteration that nothing it did caused the failure. With `harness_error=False`
    harbor POSITIVELY recorded no exception, so claiming a harness fault is the
    stronger unevidenced claim -- and it is the claim whose failure mode is a loop
    that never improves, because the agent is told to change nothing and stalls
    again identically.

    Crucially this removes the asymmetry the defect describes: telemetry_absent with
    no harness error now classifies the SAME way whether or not `finished_at`
    happened to be flushed before teardown. A missing timestamp no longer flips the
    blame.
    """
    assert classify(0.0, "telemetry_absent", 1,
                    harness_error=False) == "agent_abandoned_run"


def test_unknown_budget_with_harness_error_is_still_the_harness():
    """A recorded harbor exception is real evidence and still outranks everything."""
    assert classify(0.0, "telemetry_absent", 1, harness_error=True,
                    budget_used_frac=None) == "harness_incomplete"


def test_a_missing_timestamp_does_not_change_the_verdict():
    """DEFECT 2 regression: same root cause -> same verdict, flushed or not."""
    flushed = classify(0.0, "telemetry_absent", 1, harness_error=False,
                       budget_used_frac=0.3)
    unflushed = classify(0.0, "telemetry_absent", 1, harness_error=False,
                         budget_used_frac=None)
    assert flushed == unflushed == "agent_abandoned_run"


@pytest.mark.parametrize("frac,expected", [
    (0.0, "agent_abandoned_run"),
    (0.3, "agent_abandoned_run"),
    (0.79, "agent_abandoned_run"),
    (0.8, "harness_incomplete"),
    (0.9, "harness_incomplete"),
    (1.0, "harness_incomplete"),
])
def test_known_budget_behaviour_is_untouched(frac, expected):
    """DEFECT 2 must change ONLY the unknown-budget case."""
    assert classify(0.0, "telemetry_absent", 1, harness_error=False,
                    budget_used_frac=frac) == expected


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


# --- DEFECT 3: "unknown" must be observable, not silent -----------------


def test_unrecognised_reason_warns_on_stderr_quoting_it_verbatim(capsys):
    """A reason nobody taught us about must surface, not vanish into "unknown".

    history.py fires no guidance block for "unknown", so the agent is told nothing
    about why it scored 0.0. These substrings were reverse-engineered from a
    reference pipeline and have never been validated against this harness's live
    verifier, so the first real run has to be able to shout the string it saw.
    """
    assert classify(0.0, "quantiser_desync[42]", 2) == "unknown"
    err = capsys.readouterr().err
    assert "quantiser_desync[42]" in err
    assert "unknown" in err.lower()


def test_empty_reason_does_not_warn(capsys):
    """No reason string is not an UNRECOGNISED reason string -- nothing to report."""
    assert classify(0.0, "", 2) == "unknown"
    assert capsys.readouterr().err == ""


def test_none_reason_does_not_warn(capsys):
    assert classify(0.0, None, 2) == "unknown"
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("reason,outcome", [
    ("no_step_clears", "graded_miss"),
    ("chain_mismatch", "gate_fail"),
    ("telemetry_absent", "agent_abandoned_run"),
])
def test_recognised_reasons_never_warn(reason, outcome, capsys):
    assert classify(0.0, reason, 2, harness_error=False,
                    budget_used_frac=None) == outcome
    assert capsys.readouterr().err == ""


def test_graded_pass_with_an_odd_reason_does_not_warn(capsys):
    """A scored run is not an unrecognised-reason problem."""
    assert classify(0.7, "quantiser_desync", 2) == "graded_pass"
    assert capsys.readouterr().err == ""


# --- DEFECT 3: the token list must stay honest --------------------------


def test_recognised_reason_tokens_is_exported_and_non_empty():
    assert isinstance(RECOGNISED_REASON_TOKENS, (frozenset, set, tuple))
    assert RECOGNISED_REASON_TOKENS


def test_recognised_reason_tokens_matches_the_branches_actually_tested():
    """Fails if someone adds an `X in r` branch without updating the constant.

    Read off the source rather than restated by hand, so the constant cannot drift
    away from the `if` statements it documents.
    """
    src = inspect.getsource(classify_mod.classify)
    in_branch = set(re.findall(r'["\']([^"\']+)["\']\s+in\s+r\b', src))
    assert in_branch == set(RECOGNISED_REASON_TOKENS), (
        f"branches test {sorted(in_branch)} but "
        f"RECOGNISED_REASON_TOKENS lists {sorted(RECOGNISED_REASON_TOKENS)}"
    )


@pytest.mark.parametrize("token", ["telemetry_absent", "seed", "not_bound",
                                   "no_step_clears", "frozen", "reconcil", "chain"])
def test_every_documented_token_is_listed(token):
    assert token in RECOGNISED_REASON_TOKENS


@pytest.mark.parametrize("token", ["telemetry_absent", "seed", "not_bound",
                                   "no_step_clears", "frozen", "reconcil", "chain"])
def test_no_listed_token_ever_reaches_unknown(token, capsys):
    """The constant is only meaningful if membership really does route somewhere."""
    assert classify(0.0, token, 2) != "unknown"
    assert capsys.readouterr().err == ""
