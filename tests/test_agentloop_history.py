"""Tests for agentloop.history -- the feedback renderer whose output is written to a
markdown file and injected verbatim into the NEXT iteration's agent prompt.

This is the highest-consequence module in the refinement loop, and the failure
modes that matter are not "does it produce text":

  * LEAKAGE. If grader vocabulary (rubric names, checker ids, the grading script)
    reaches the agent, the agent can optimise the rubric instead of the task. The
    `scrub` firewall is therefore tested both directly AND end-to-end through the
    full renderer, because a leak that bypasses `scrub` on any path is a leak.
  * MIS-REPORTING. An infrastructure failure carries reward 0.0 in the row. Printed
    as `0.0000` it reads as "your optimizer scored zero" and teaches the agent to
    abandon a rule that was never actually graded. Only `graded_pass`/`graded_miss`
    rows may show a number; everything else must say `not graded`.
  * FABRICATION. `--backend dry` invents a val_loss curve from a seed hash. Those
    rows must be banner-marked at the TOP of the history or they read as real
    training results.
  * ANCHORING. A parent optimizer's source is carried forward only when the
    verifier actually scored it above zero; an unscored parent is an unvalidated
    guess that anchors the next attempt onto code nothing supports.

No test here touches the network, the filesystem under track3-pipeline/, or any
trial directory: the renderer is a pure function of ledger rows.
"""

import re

import pytest

from agentloop.history import (
    FORBIDDEN,
    MAX_HISTORY_CHARS,
    MAX_HISTORY_ITERS,
    _restate_budget,
    render_history,
    render_parent,
    scrub,
)
from agentloop.marking import SYNTHETIC_BANNER

FACTS_HEADER = "| iter | reward | graded_step | outcome (classified) | seeds |"

# One live example per FORBIDDEN pattern, in pattern order.
LEAKY = ("rubric J3 checker_foo C_ABCDE NOVELTY_FLOOR reference_optimizer "
         "corpus_manifest TRACK3_OUTCOMES conformance.py grade.py")


def row(iteration=0, reward=0.0, outcome="graded_miss", **kw):
    r = {"iteration": iteration, "reward": reward, "outcome": outcome,
         "n_seeds": 2, "graded_step": 3000, "findings": "", "summary": {}}
    r.update(kw)
    return r


def assert_no_forbidden(text):
    """Not one of the ten patterns may survive anywhere in `text`."""
    leaks = [p for p in FORBIDDEN if re.search(p, text, re.IGNORECASE)]
    assert leaks == [], f"grader vocabulary leaked to the agent: {leaks}"


# --------------------------------------------------------------------------
# constants -- pinned, because they are the prompt's size contract
# --------------------------------------------------------------------------

def test_constants_preserved():
    from agentloop import history
    assert history.MAX_HISTORY_ITERS == 8
    assert history.MAX_FINDINGS_CHARS == 1200
    assert history.MAX_HISTORY_CHARS == 30000
    assert history.MAX_PARENT_SOURCE_CHARS == 18000
    assert history.MAX_CURVE_POINTS == 12


def test_forbidden_is_the_ten_patterns():
    assert tuple(FORBIDDEN) == (
        r"\brubric",
        r"\bJ[1-8]\b",
        r"\bchecker_[a-z_]+",
        r"\bC_[A-Z_]{4,}",
        r"NOVELTY_FLOOR",
        r"reference_optimizer",
        r"\bcorpus_manifest",
        r"TRACK3_OUTCOMES",
        r"conformance\.py",
        r"\bgrade\.py",
    )


# --------------------------------------------------------------------------
# the scrub firewall
# --------------------------------------------------------------------------

def test_scrub_removes_every_forbidden_pattern():
    out, hits = scrub(LEAKY)
    assert_no_forbidden(out)
    assert "[redacted]" in out
    assert hits, "redactions must be reported, not silently swallowed"
    assert hits == sorted(set(hits)), "hits must be sorted and de-duplicated"
    assert all(isinstance(h, str) for h in hits)


def test_scrub_is_case_insensitive():
    out, hits = scrub("RUBRIC and Conformance.PY and Novelty_Floor")
    assert_no_forbidden(out)
    assert "RUBRIC" not in out
    assert hits


def test_scrub_leaves_innocent_text_alone():
    clean = "Used Muon with lr 0.02 on 2 seeds; val_loss 3.28 at step 3000."
    out, hits = scrub(clean)
    assert out == clean
    assert hits == []


def test_scrub_returns_str_hits_for_grouped_patterns():
    # \bJ[1-8]\b has no group, but findall on some patterns yields tuples upstream;
    # every hit must be coerced to str so sorted() cannot raise.
    _, hits = scrub("J1 J2 J1 checker_alpha checker_alpha")
    assert hits == sorted(set(hits))
    assert all(isinstance(h, str) for h in hits)


def test_render_history_scrubs_end_to_end(capsys):
    out = render_history([row(iteration=0, reward=0.4, outcome="graded_pass",
                              findings=f"I read the {LEAKY} and tuned to it.")])
    assert_no_forbidden(out)
    assert "[redacted]" in out
    assert "[scrub]" in capsys.readouterr().err


def test_render_history_scrubs_summary_prose_too():
    summary = {"mechanism": f"followed {LEAKY}", "measurements": "val_loss 3.1"}
    out = render_history([row(iteration=1, reward=0.4, outcome="graded_pass",
                              summary=summary)])
    assert_no_forbidden(out)


# --------------------------------------------------------------------------
# "not graded" -- an ungraded run is not a zero score
# --------------------------------------------------------------------------

def test_ungraded_row_never_shows_a_number():
    out = render_history([row(iteration=0, reward=0.0,
                              outcome="harness_incomplete")])
    assert "not graded" in out
    assert "0.0000" not in out


@pytest.mark.parametrize("outcome", ["agent_abandoned_run", "harness_incomplete",
                                     "gate_fail", "unknown"])
def test_every_ungraded_outcome_says_not_graded(outcome):
    out = render_history([row(iteration=0, reward=0.0, outcome=outcome)])
    assert "not graded" in out
    assert "0.0000" not in out


def test_graded_miss_at_zero_does_show_the_number():
    out = render_history([row(iteration=0, reward=0.0, outcome="graded_miss")])
    assert "0.0000" in out
    assert "not graded" not in out


def test_facts_table_present():
    out = render_history([row()])
    assert FACTS_HEADER in out


# --------------------------------------------------------------------------
# steering: exploit / explore / nothing-graded
# --------------------------------------------------------------------------

def test_exploit_block_quotes_the_best_reward():
    rows = [row(iteration=0, reward=0.10, outcome="graded_pass"),
            row(iteration=1, reward=0.58, outcome="graded_pass"),
            row(iteration=2, reward=0.20, outcome="graded_pass")]
    out = render_history(rows)
    assert "Beat 0.5800" in out
    assert "EXPLOIT" in out


def test_explore_block_when_scored_but_all_zero():
    rows = [row(iteration=0, reward=0.0, outcome="graded_miss"),
            row(iteration=1, reward=0.0, outcome="harness_incomplete")]
    out = render_history(rows)
    assert "EXPLORE" in out
    assert "Beat" not in out


def test_nothing_graded_block_when_no_row_scored():
    rows = [row(iteration=0, reward=0.0, outcome="harness_incomplete"),
            row(iteration=1, reward=0.0, outcome="agent_abandoned_run")]
    out = render_history(rows)
    assert "EXPLORE" not in out
    assert "Beat" not in out
    assert "no evidence for or against" in out


def test_abandoned_last_run_gets_its_own_warning():
    out = render_history([row(iteration=0, reward=0.0,
                              outcome="agent_abandoned_run")])
    assert "never graded" in out
    assert "## Telemetry binding" in out


def test_no_telemetry_block_for_a_clean_graded_history():
    out = render_history([row(iteration=0, reward=0.4, outcome="graded_pass")])
    assert "## Telemetry binding" not in out


# --------------------------------------------------------------------------
# `unknown` -- the outcome that used to get NO guidance at all
# --------------------------------------------------------------------------
#
# `classify` returns "unknown" whenever the verifier emits a reason string it
# does not recognise. Before this block existed the agent saw outcome `unknown`,
# reward cell `not graded`, and nothing else: no account of why it scored 0.0
# and no instruction on what to do differently. That is the live pipeline's
# actual failure mode -- three consecutive unscored iterations -- so `unknown`
# must steer at least as hard as `agent_abandoned_run` does.

UNKNOWN_MARKER = "no recognisable verdict"


def test_unknown_last_run_gets_guidance():
    out = render_history([row(iteration=0, reward=0.0, outcome="unknown")])
    assert UNKNOWN_MARKER in out


def test_unknown_guidance_says_the_zero_is_not_a_refutation():
    """The whole point: a 0.0 from an unread verdict is not evidence against the rule."""
    out = render_history([row(iteration=0, reward=0.0, outcome="unknown")])
    assert "NOT evidence" in out
    assert "number on the board" in out


def test_unknown_row_gets_the_telemetry_block():
    """An unrecognised verdict most often means the telemetry never bound."""
    out = render_history([row(iteration=0, reward=0.0, outcome="unknown")])
    assert "## Telemetry binding" in out


def test_graded_pass_does_not_get_the_unknown_guidance():
    out = render_history([row(iteration=0, reward=0.4, outcome="graded_pass")])
    assert UNKNOWN_MARKER not in out


@pytest.mark.parametrize("outcome", ["graded_pass", "graded_miss",
                                     "harness_incomplete", "agent_abandoned_run"])
def test_unknown_guidance_fires_for_no_other_outcome(outcome):
    out = render_history([row(iteration=0, reward=0.0, outcome=outcome)])
    assert UNKNOWN_MARKER not in out


def test_unknown_guidance_follows_the_last_row_not_an_older_one():
    """Same gating as the abandoned-run block: the last attempt is the actionable one."""
    rows = [row(iteration=0, reward=0.0, outcome="unknown"),
            row(iteration=1, reward=0.4, outcome="graded_pass")]
    assert UNKNOWN_MARKER not in render_history(rows)
    rows = [row(iteration=0, reward=0.4, outcome="graded_pass"),
            row(iteration=1, reward=0.0, outcome="unknown")]
    assert UNKNOWN_MARKER in render_history(rows)


def test_unknown_guidance_leaks_no_grader_vocabulary():
    out = render_history([row(iteration=0, reward=0.0, outcome="unknown")])
    assert_no_forbidden(out)


def test_unknown_guidance_does_not_break_the_length_trim():
    """The new block is prose the trim cannot drop, so it must not push past the cap."""
    rows = [row(iteration=i, reward=0.0, outcome="graded_miss",
                findings=f"iteration {i} account. " + ("blah " * 1200))
            for i in range(30)]
    rows[-1] = row(iteration=29, reward=0.0, outcome="unknown",
                   findings="last account. " + ("blah " * 1200))
    out = render_history(rows)
    assert len(out) <= MAX_HISTORY_CHARS, f"{len(out)} > {MAX_HISTORY_CHARS}"
    assert FACTS_HEADER in out
    assert UNKNOWN_MARKER in out
    assert "## Telemetry binding" in out
    for i in range(22, 30):
        assert f"| {i} | " in out


def test_unknown_guidance_is_short_enough_to_leave_room_for_the_facts():
    """A guidance block the trim cannot sacrifice must stay a small slice of budget."""
    base = len(render_history([row(iteration=0, reward=0.0, outcome="harness_incomplete")]))
    grown = len(render_history([row(iteration=0, reward=0.0, outcome="unknown")]))
    assert grown - base < MAX_HISTORY_CHARS // 20


# --------------------------------------------------------------------------
# header prose
# --------------------------------------------------------------------------

def test_header_states_attempt_number_and_remaining_attempts():
    # the attempt number comes from the last row's iteration + 1, not from len(rows)
    rows = [row(iteration=i) for i in range(3)]
    out = render_history(rows, total_iterations=5)
    assert "# Attempt 3" in out
    assert "3 times" in out
    assert "2 further attempts will follow" in out


def test_header_uses_the_singular_for_one_remaining_attempt():
    out = render_history([row(iteration=i) for i in range(3)], total_iterations=4)
    assert "1 further attempt will follow" in out


def test_header_is_noncommittal_without_a_total():
    assert "further attempts may follow" in render_history([row()])


def test_header_says_last_attempt_when_budget_exhausted():
    out = render_history([row(iteration=i) for i in range(4)], total_iterations=4)
    assert "this is your last attempt" in out


def test_header_singular_once_for_a_single_prior_attempt():
    out = render_history([row(iteration=0)])
    assert "once" in out


def test_header_quotes_the_current_budget():
    out = render_history([row()], budget_hours="8 hours of wall clock")
    assert "CURRENT budget is 8 hours of wall clock" in out


# --------------------------------------------------------------------------
# parent carry
# --------------------------------------------------------------------------

def test_render_parent_empty_when_unscored():
    carried = row(iteration=2, reward=0.0, outcome="harness_incomplete",
                  parent_source="import torch\n")
    assert render_parent(carried) == ""


def test_render_parent_empty_without_source():
    assert render_parent(row(iteration=2, reward=0.9)) == ""


def test_render_parent_emits_fenced_source_when_scored():
    carried = row(iteration=2, reward=0.4, outcome="graded_pass",
                  parent_source="import torch  # PARENT_BODY\n")
    out = render_parent(carried)
    assert "## Your previous submission (iteration 2)" in out
    assert "```python" in out
    assert "PARENT_BODY" in out


def test_render_parent_notes_truncation_and_curve():
    carried = row(iteration=1, reward=0.4, outcome="graded_pass",
                  parent_source="x = 1", parent_source_truncated=True,
                  parent_curve={"0": [(100, 4.5), (200, 4.0)]})
    out = render_parent(carried)
    assert "(truncated)" in out
    assert "seed0  100:4.50000  200:4.00000" in out


def test_history_carries_parent_of_the_best_scored_row():
    rows = [row(iteration=0, reward=0.1, outcome="graded_pass",
                parent_source="# LOSER"),
            row(iteration=1, reward=0.6, outcome="graded_pass",
                parent_source="# WINNER")]
    out = render_history(rows)
    assert "# WINNER" in out
    assert "# LOSER" not in out


def test_parent_tie_break_prefers_the_later_iteration():
    rows = [row(iteration=0, reward=0.5, outcome="graded_pass",
                parent_source="# EARLIER"),
            row(iteration=1, reward=0.5, outcome="graded_pass",
                parent_source="# LATER")]
    out = render_history(rows)
    assert "# LATER" in out
    assert "# EARLIER" not in out
    assert "## Your previous submission (iteration 1)" in out


def test_history_carries_no_parent_when_best_scored_zero():
    rows = [row(iteration=0, reward=0.0, outcome="graded_miss",
                parent_source="# UNVALIDATED")]
    assert "# UNVALIDATED" not in render_history(rows)


# --------------------------------------------------------------------------
# synthetic marking
# --------------------------------------------------------------------------

def test_synthetic_banner_appears_near_the_top():
    rows = [row(iteration=0, reward=0.3, outcome="graded_pass", backend="dry")]
    out = render_history(rows)
    assert SYNTHETIC_BANNER in out
    assert out.index(SYNTHETIC_BANNER) < out.index(FACTS_HEADER)


def test_synthetic_flag_alone_triggers_the_banner():
    rows = [row(iteration=0, reward=0.3, outcome="graded_pass",
                backend="harbor", is_synthetic=True)]
    assert SYNTHETIC_BANNER in render_history(rows)


def test_no_banner_for_real_rows():
    rows = [row(iteration=0, reward=0.3, outcome="graded_pass", backend="harbor")]
    assert SYNTHETIC_BANNER not in render_history(rows)


def test_synthetic_banner_survives_trimming():
    rows = [row(iteration=i, reward=0.1, outcome="graded_pass", backend="dry",
                findings="x " * 4000) for i in range(12)]
    out = render_history(rows)
    assert SYNTHETIC_BANNER in out
    assert FACTS_HEADER in out


# --------------------------------------------------------------------------
# windowing and trimming
# --------------------------------------------------------------------------

def test_only_the_last_max_history_iters_appear():
    rows = [row(iteration=i, reward=0.0, outcome="graded_miss") for i in range(12)]
    out = render_history(rows)
    assert MAX_HISTORY_ITERS == 8
    for i in range(4):                      # 0..3 fall outside the window
        assert f"| {i} | " not in out
    for i in range(4, 12):
        assert f"| {i} | " in out


def test_trim_bounds_the_prompt_and_keeps_the_facts_table():
    parent = "# PARENT\n" + ("y = 1\n" * 500)
    rows = [row(iteration=i, reward=0.1 + i * 0.01, outcome="graded_pass",
                findings=f"iteration {i} account. " + ("blah " * 1200))
            for i in range(30)]
    rows[-1]["parent_source"] = parent
    out = render_history(rows)
    bound = MAX_HISTORY_CHARS + len(render_parent(rows[-1]))
    assert len(out) <= bound, f"{len(out)} > {bound}"
    assert FACTS_HEADER in out
    assert "## Now attempt the task again" in out
    assert "# PARENT" in out
    # the facts table still reports all eight windowed iterations
    for i in range(22, 30):
        assert f"| {i} | " in out
    # the oldest per-iteration prose is what was sacrificed
    assert "## Iteration 22 — what you tried" not in out
    assert "## Iteration 29 — what you tried" in out


# --------------------------------------------------------------------------
# budget restatement and the empty case
# --------------------------------------------------------------------------

def test_restate_budget_rewrites_stale_hour_figures():
    assert "8-hour" in _restate_budget("a 5-hour cap", "8 hours of wall clock")
    assert "8-hour" in _restate_budget("a 4.5 hour cap", "8 hours of wall clock")
    assert "8-hour" in _restate_budget("I had 6 hour", "8 hours of wall clock")


def test_restate_budget_does_not_rewrite_plural_hours():
    """Pins a KNOWN GAP in the ported regex rather than hiding it.

    `\\b\\d+(?:\\.\\d+)?\\s*-?\\s*hour\\b` cannot match "6 hours": the trailing `\\b`
    fails against the `s`. So a summary saying "I had 6 hours" is forwarded with its
    stale figure intact. This is upstream behaviour, reproduced deliberately; the test
    exists so that widening the regex to `hours?` is a conscious edit with a failing
    test to update, not a silent behaviour change in agent-facing prose.
    """
    assert _restate_budget("You have 6 hours", "8 hours of wall clock") == "You have 6 hours"


def test_restate_budget_leaves_other_numbers_alone():
    out = _restate_budget("3000 steps at lr 0.02", "8 hours of wall clock")
    assert out == "3000 steps at lr 0.02"


def test_restate_budget_applied_to_forwarded_prose():
    out = render_history([row(iteration=0, reward=0.4, outcome="graded_pass",
                              findings="I planned against a 5-hour budget.")],
                         budget_hours="8 hours of wall clock")
    assert "8-hour budget" in out
    assert "5-hour" not in out


def test_empty_history_renders_nothing():
    assert render_history([]) == ""
