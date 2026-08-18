"""Outcome classification for track3 iterations.

Ported from track3-pipeline/tools/refine.py (classify, _graded_step).
"""
import sys


RECOGNISED_REASON_TOKENS = frozenset({
    "telemetry_absent",
    "seed",
    "not_bound",
    "no_step_clears",
    "frozen",
    "reconcil",
    "chain",
})


def classify(reward: float, reason: str, n_seeds: int,
             harness_error: bool = False, budget_used_frac: float | None = None) -> str:
    """Separate the three distinct ways a run yields 0.0.

    agent_abandoned_run is the case the packaged r4 record mislabelled as a harness fault:
    Harbor logged no exception, 68% of the agent budget was unspent, and the trial ended
    only because the agent stopped producing foreground work while its graded run was still
    training. Calling that a harness failure teaches the next iteration to blame the harness
    for a choice it made itself.

    An UNMEASURABLE budget is treated the same way. `_budget_used` returns None when
    the container was torn down before harbor flushed `finished_at`, which is not
    evidence that the budget was spent -- yet the old code let that missing timestamp
    flip the verdict, so one root cause produced two labels. `harness_incomplete` is
    not a neutral don't-know bucket: it affirmatively blames the harness and is the
    label under which the next iteration is told to change nothing. With
    harness_error False, harbor POSITIVELY recorded no exception, so that is the less
    defensible default and the one whose failure mode is a loop that stalls forever.
    A known-spent budget still yields harness_incomplete; only the unknown case moved.

    The reason branches below are the ONLY place these tokens are tested, and their
    order is load-bearing. RECOGNISED_REASON_TOKENS mirrors them so the coupling is
    discoverable, and a fall-through to unknown on a non-empty reason is announced on
    stderr rather than swallowed, because no guidance block downstream fires for
    unknown and the agent would learn nothing about why it scored 0.0.
    """
    if reward and reward > 0:
        return "graded_pass"
    r = (reason or "").lower()
    ungradable = (n_seeds < 2 or "telemetry_absent" in r or "seed" in r
                  or "not_bound" in r)
    if ungradable:
        spent = budget_used_frac is not None and budget_used_frac >= 0.8
        if not harness_error and not spent:
            return "agent_abandoned_run"
        return "harness_incomplete"
    if "no_step_clears" in r:
        return "graded_miss"
    if "frozen" in r or "reconcil" in r or "chain" in r:
        return "gate_fail"
    if r:
        print(f"WARNING: track3.classify: UNRECOGNISED verifier reason {reason!r} "
              f"-> outcome 'unknown'. No guidance block fires for 'unknown', so the "
              f"next iteration will be told nothing about this failure. Add the token "
              f"to RECOGNISED_REASON_TOKENS and to a branch in classify(). "
              f"Currently recognised: {sorted(RECOGNISED_REASON_TOKENS)}",
              file=sys.stderr)
    return "unknown"


def _graded_step(reason: str):
    if reason and reason.startswith("graded_step="):
        try:
            return int(reason.split("=", 1)[1])
        except ValueError:
            return None
    return None
