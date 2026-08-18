"""Verifier for bia/minicalc. Stdlib only; runs in the agent's own container.

This is the miniature of the bia/track3nov grader, and it is deliberately the SAME
scoring rule as `runner/agentloop/reward.py`, not a superset of it:

    graded_step = max over seeds of (first step whose val_loss <= TARGET_LOSS)
    reward      = clip((BASELINE_STEPS - graded_step) / (BASELINE_STEPS - TARGET_STEPS), 0, 1)

The real bundle's grader adds a significance margin and a persistence check that
`reward.py` does not reproduce, so the two disagree by construction there. Here they
must agree EXACTLY, because agreeing is the thing this task exists to demonstrate:
the loop's score and the task's score are computed by two independent
implementations of one rule, and a divergence is a defect in one of them.

AGREEMENT ACROSS SEEDS is the same rule as the real task's: every seed must reach
the target on its own, and the run is only as fast as its slowest seed. A mean-only
reading would let one lucky seed carry a failing one.

REASON STRINGS ARE PART OF THE CONTRACT. `runner/agentloop/classify.py` turns
`reason` into an outcome by substring match against RECOGNISED_REASON_TOKENS, and
a string it does not recognise becomes 'unknown', for which no guidance block fires
and the next iteration is told nothing. Every reason emitted below therefore carries
one of those tokens:

    reward > 0            -> "graded_step=<n>"            -> graded_pass
    cleared, but too late -> "...no_step_clears..."        -> graded_miss
    a seed never cleared  -> "no_step_clears_target_loss"  -> graded_miss
    fewer than 2 seeds    -> "...seeds..." (token "seed")  -> agent_abandoned_run
    no parsable logs      -> "telemetry_absent"            -> agent_abandoned_run
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

BASELINE_STEPS = 3500
TARGET_STEPS = 2900
TARGET_LOSS = 3.28
MIN_SEEDS = 2

# Byte-for-byte the pattern in runner/agentloop/trial_io.py. If the trainer's log
# format ever drifts from it, both scorers go silent together rather than one of
# them inventing a number.
VAL_LOSS_RE = re.compile(r"step:(\d+)/\d+ val_loss:([\d.]+)")

BUNDLE = os.environ.get("MINICALC_BUNDLE", "/workspace")
SUBMISSION = os.environ.get("MINICALC_SUBMISSION", os.path.join(BUNDLE, "submission"))
REWARD_PATH = os.environ.get("REWARD_PATH", os.path.join(BUNDLE, "reward.json"))


def load_seed_logs(submission: str) -> dict[str, dict[int, float]]:
    """`{seed: {step: loss}}` for every `full_seed*.log` that parsed at all.

    A log that exists but yields no points is dropped rather than recorded empty:
    a seed with no measurement did not run, and counting it would make `n_seeds`
    claim a coverage the run does not have.
    """
    out: dict[str, dict[int, float]] = {}
    for path in sorted(glob.glob(os.path.join(submission, "logs", "full_seed*.log"))):
        seed = os.path.basename(path)[len("full_seed"):].split(".")[0]
        try:
            with open(path, errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        points = {int(m.group(1)): float(m.group(2))
                  for m in VAL_LOSS_RE.finditer(text)}
        if points:
            out[seed] = points
    return out


def graded_step(seed_logs) -> tuple[int | None, str | None, dict]:
    """First step by which EVERY seed has reached TARGET_LOSS, or `(None, reason, ...)`.

    THE REASON MUST NOT NAME THE SEED. `classify()` tests its tokens in order and
    'seed' is in the FIRST branch, the ungradable one -- so a perfectly gradable miss
    reported as `no_step_clears_target_loss_seed_0` is classified 'harness_incomplete'
    or 'agent_abandoned_run' instead of 'graded_miss', and the next iteration is told
    the harness broke when in fact its optimizer was too slow. Which seed failed goes
    in `detail`, where no classifier reads it.
    """
    if len(seed_logs) < MIN_SEEDS:
        return None, f"need_at_least_{MIN_SEEDS}_seeds_got_{len(seed_logs)}", {}
    crossings = {}
    for seed, points in sorted(seed_logs.items()):
        first = min((s for s, v in points.items() if v <= TARGET_LOSS), default=None)
        if first is None:
            return None, "no_step_clears_target_loss", {"failing_seed": seed}
        crossings[seed] = first
    return max(crossings.values()), None, {"first_crossing": crossings}


def emit(reward: float, reason: str, detail: dict, n_seeds: int) -> float:
    record = {
        "reward": reward,
        "reason": reason,
        "detail": detail,
        "metrics": {"n_seeds": n_seeds},
    }
    os.makedirs(os.path.dirname(REWARD_PATH) or ".", exist_ok=True)
    with open(REWARD_PATH, "w") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    print(json.dumps(record, sort_keys=True))
    return reward


def main() -> float:
    seed_logs = load_seed_logs(SUBMISSION)
    n_seeds = len(seed_logs)
    if not seed_logs:
        return emit(0.0, "telemetry_absent", {"submission": SUBMISSION}, 0)

    step, why, extra = graded_step(seed_logs)
    detail = {
        "seeds": sorted(seed_logs),
        "final_loss": {s: p[max(p)] for s, p in sorted(seed_logs.items())},
        "baseline_steps": BASELINE_STEPS,
        "target_steps": TARGET_STEPS,
        "target_loss": TARGET_LOSS,
        **extra,
    }
    if step is None:
        return emit(0.0, why or "no_step_clears_target_loss", detail, n_seeds)

    detail["graded_step"] = step
    reward = max(0.0, min(1.0, (BASELINE_STEPS - step)
                          / (BASELINE_STEPS - TARGET_STEPS)))
    if reward <= 0.0:
        # Cleared the loss, but not before the baseline. This is a MISS, not an
        # absence, and must not be reported as `graded_step=` -- classify() reads
        # that token as a pass and there is no reward to justify one.
        return emit(0.0, f"no_step_clears_baseline_reached_target_at_step_{step}",
                    detail, n_seeds)
    return emit(reward, f"graded_step={step}", detail, n_seeds)


if __name__ == "__main__":
    main()
    sys.exit(0)
