"""Outcome classification for track3 iterations.

Ported verbatim from track3-pipeline/tools/refine.py (classify, _graded_step).
"""


def classify(reward: float, reason: str, n_seeds: int,
             harness_error: bool = False, budget_used_frac: float | None = None) -> str:
    """Separate the three distinct ways a run yields 0.0.

    agent_abandoned_run is the case the packaged r4 record mislabelled as a harness fault:
    Harbor logged no exception, 68% of the agent budget was unspent, and the trial ended
    only because the agent stopped producing foreground work while its graded run was still
    training. Calling that a harness failure teaches the next iteration to blame the harness
    for a choice it made itself.
    """
    if reward and reward > 0:
        return "graded_pass"
    r = (reason or "").lower()
    ungradable = (n_seeds < 2 or "telemetry_absent" in r or "seed" in r
                  or "not_bound" in r)
    if ungradable:
        unspent = budget_used_frac is not None and budget_used_frac < 0.8
        if not harness_error and unspent:
            return "agent_abandoned_run"
        return "harness_incomplete"
    if "no_step_clears" in r:
        return "graded_miss"
    if "frozen" in r or "reconcil" in r or "chain" in r:
        return "gate_fail"
    return "unknown"


def _graded_step(reason: str):
    if reason and reason.startswith("graded_step="):
        try:
            return int(reason.split("=", 1)[1])
        except ValueError:
            return None
    return None
