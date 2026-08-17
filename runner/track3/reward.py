"""Reward from the measured loss curve, independent of the task verifier.

The verifier (`verifier/reward_full.json`) and the LLM judge are currently UNWIRED
from the loop -- see the module note in `track3/loop.py`. This module is what takes
their place: it turns the val_loss curve the trial actually wrote into the single
number the ledger ranks iterations by.

    reward = max(0.0, min(1.0, (BASELINE_STEPS - step) / (BASELINE_STEPS - TARGET_STEPS)))

Three decisions here are load-bearing.

THE CONSTANTS ARE MIRRORED, NOT IMPORTED. `BASELINE_STEPS`, `TARGET_STEPS` and
`TARGET_LOSS` come from `tasks/<uuid>/tests/grade.py` (lines 34-36), which lives
inside the task bundle and is executed in-container against a stdlib-only interpreter.
Importing it from here would couple the outer loop to a task's test tree; mirroring
costs a comment and keeps this module usable when no task bundle is on the path.

AGREEMENT ACROSS SEEDS. `grade.py::graded_step` documents the rule this implements:
"every seed must individually reach TARGET_LOSS. instruction.md promises 'at least
two seeds must clear'; a mean-only test lets one lucky seed carry a failing one."
So `step_to_target` takes the MAX of the per-seed first crossings -- the run has not
reached target until its SLOWEST seed has -- and returns None the moment any seed
fails to cross at all.

ABSENCE IS DATA. Every helper here is fed by `trial_io.parent_artifacts`, which parses
logs from runs that may have crashed, been killed mid-write, or never trained. A
missing curve, an empty seed, or a malformed point yields None/0.0 rather than an
exception, because a parser that raises on a partial trial turns a recoverable
iteration into a dead loop.

DENSITY IS NOT A PARAMETER OF THE SCORE. `trial_io` feeds these helpers the
FULL-DENSITY parsed log points (`full_curve`), never the thinned `parent_curve`.
Thinning exists only to keep the agent-facing history markdown small; it can drop the
first point that crosses the target and so can only ever report a crossing late or
not at all. Scoring off it would make `trial_io.MAX_CURVE_POINTS` -- a display
constant -- silently re-score every run in the campaign.

DIVERGENCE FROM grade.py. The in-container grader additionally enforces a
significance MARGIN (SIG_MARGIN/sqrt(n)) and PERSISTENCE (the crossing must not be
given back at any later logged step). Neither is reproduced here, deliberately: this
is the loop's own steering signal for ranking iterations against each other, not the
task's grade, and first-crossing-per-seed is the comparable quantity across runs.
"""
from __future__ import annotations

# Mirrored from tasks/2739a678-1759-516d-8ba7-1cd023267ea8/tests/grade.py:34-36.
BASELINE_STEPS = 3500
TARGET_STEPS = 2900
TARGET_LOSS = 3.28


def compute_reward(step: int | float | None) -> float:
    """Linear credit for reaching the target loss `step` steps in, clamped to [0, 1].

    3500 -> 0.0, 3200 -> 0.5, 2900 -> 1.0. `None` means the target was never reached,
    which is 0.0 -- not a small reward but no reward, so `classify` can tell a graded
    miss from a run that produced no measurement at all.
    """
    if step is None:
        return 0.0
    return max(0.0, min(1.0,
                        (BASELINE_STEPS - step) / (BASELINE_STEPS - TARGET_STEPS)))


def step_to_target(curve: dict[str, list], target_loss: float = TARGET_LOSS) -> int | None:
    """The step at which EVERY seed has reached `target_loss`, or None.

    Per seed: the FIRST step whose val_loss is at or below the target. Across seeds:
    the MAX of those, because the run only counts as having reached the target once
    the slowest seed does. If any seed never reaches it the run is ungraded (None),
    which is `grade.py`'s AGREEMENT rule -- a mean-only or min-only reading would let
    one lucky seed carry a failing one.
    """
    if not curve:
        return None
    crossings = []
    for points in curve.values():
        first = None
        for point in points or ():
            try:
                step, val = int(point[0]), float(point[1])
            except (TypeError, ValueError, IndexError, KeyError):
                # A truncated or malformed line in a partial log is skipped, not fatal.
                continue
            if val <= target_loss and (first is None or step < first):
                first = step
        if first is None:
            # This seed never got there, so neither did the run.
            return None
        crossings.append(first)
    return max(crossings) if crossings else None


def reward_from_curve(curve, target_loss: float = TARGET_LOSS) -> tuple[float, int | None]:
    """`(reward, step)` for a `parent_curve`-shaped dict of `{seed: [[step, loss], ...]}`.

    Returned as a pair because the ledger records both: the step is the fact, and the
    reward is the ranking of it. A caller that stored only the reward could not tell
    0.0-because-never-crossed from 0.0-because-crossed-at-the-baseline.
    """
    step = step_to_target(curve, target_loss)
    return compute_reward(step), step
