"""Read one completed Harbor trial directory into a flat ledger row.

Ported from track3-pipeline/tools/refine.py (`read_trial`, `parent_artifacts`,
`_budget_used`, `agent_findings`, `find_trial`, `_task_budget_secs`,
`_task_budget_hours`). Classification lives in the sibling `classify` module and
is imported, not duplicated.

Three things here are load-bearing and deliberate:

* The reward is COMPUTED by `agentloop.reward` from the FULL-DENSITY parsed loss curve
  (`parent_artifacts` returns it as `full_curve`), not from the thinned `parent_curve`
  that gets rendered into the agent's prompt. `MAX_CURVE_POINTS` is a display budget
  and must never move a score. `read_trial` pops `full_curve` once it has scored, so
  it never reaches the ledger row.
* The task verifier and the LLM judge are currently UNWIRED (see the note in
  `agentloop/loop.py`), so ``verifier/reward_full.json`` no longer supplies the score,
  but it is still
  read, because it is the only file carrying ``reason`` and ``metrics``. It is also
  still the FULL record and not ``verifier/reward.json``: harbor writes the short
  reward.json for its own result plumbing and the sibling judge module reads that
  file, and conflating the two silently drops the reason string, which is what
  `_graded_step` and `classify` run on.
* Every reader swallows every exception and falls back to a default. A trial dir
  is an artifact of a run that may have crashed, been killed mid-write, or never
  produced a verifier at all; a parser that raises on a partial trial turns a
  recoverable iteration into a dead loop. Absence is data — `read_trial` on an
  empty directory must return a reward-0 row, not a traceback.

DIVERGENCE FROM THE SOURCE: the original resolves the task budget through a
module-global ``TASK`` directory. This harness is multi-task, so `task_budget_secs`
and `task_budget_hours` take ``task_dir`` as a parameter and no module-level
ROOT/TASK global exists.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from datetime import datetime

from agentloop.classify import _graded_step, classify
from agentloop.reward import reward_from_curve

MAX_FINDINGS_CHARS = 1200
MAX_PARENT_SOURCE_CHARS = 18000
MAX_CURVE_POINTS = 12

# Default agent budget in seconds, used only when the trial's own config.json is
# missing or unreadable. The trial's config is authoritative because it records the
# cap the run actually operated under, which may differ from today's task.toml.
DEFAULT_BUDGET_SECS = 18000.0

VAL_LOSS_RE = re.compile(r"step:(\d+)/\d+ val_loss:([\d.]+)")


def read_trial(trial: pathlib.Path) -> dict:
    """Flatten one trial directory into the row the refinement loop records."""
    def j(p, default=None):
        try:
            return json.loads((trial / p).read_text())
        except Exception:
            return default if default is not None else {}

    rf = j("verifier/reward_full.json")
    res = j("result.json")
    agent = res.get("agent_result") or {}
    logs = sorted(trial.glob("artifacts/**/full_seed*.log"))
    metrics = rf.get("metrics") or {}
    n_seeds = int(metrics.get("n_seeds") or len(logs))
    harness_error = res.get("exception_info") is not None
    budget_frac = _budget_used(res, trial)

    # The reward is COMPUTED from the measured curve, not read from `rf["reward"]`.
    # The verifier and the LLM judge are currently unwired (see agentloop/loop.py), so
    # `agentloop.reward` is the only thing that scores an iteration. reward_full is
    # still read above and below for `reason` and `metrics`, which nothing else
    # carries, and which stay useful whenever the verifier is re-enabled.
    artifacts = parent_artifacts(trial)
    # SCORED ON THE FULL-DENSITY POINTS, NEVER ON `parent_curve`, which is thinned to
    # MAX_CURVE_POINTS purely so the history markdown fits the agent's prompt. Scoring
    # off the thinned curve would make that display constant load-bearing, and thinning
    # can only ever find a crossing late, so every retune would re-score the campaign
    # downwards. POPPED, not read: ~61 points per seed per iteration that no reader
    # downstream consumes have no business in the splat below and in every ledger row.
    curve = artifacts.pop("full_curve", None)
    reward, curve_step = reward_from_curve(curve)
    reason = rf.get("reason") or ("" if curve else "no_curve")
    # A curve that exists but never crosses may still have been graded by the verifier,
    # which applies a significance margin and a persistence check this module does not,
    # so its step is worth reporting; with no curve at all there is nothing to report
    # and None is the honest answer.
    graded_step = curve_step if curve_step is not None else (
        _graded_step(reason) if curve else None)

    return {
        "reward": reward,
        "reason": reason,
        "graded_step": graded_step,
        "n_seeds": n_seeds,
        "outcome": classify(reward, reason, n_seeds,
                            harness_error=harness_error,
                            budget_used_frac=budget_frac),
        "harness_error": harness_error,
        "budget_used_frac": budget_frac,
        "n_input_tokens": agent.get("n_input_tokens"),
        "n_output_tokens": agent.get("n_output_tokens"),
        "n_cache_tokens": agent.get("n_cache_tokens"),
        "cost_usd": agent.get("cost_usd"),
        "started_at": res.get("started_at"),
        "finished_at": res.get("finished_at"),
        "trial_dir": str(trial),
        **artifacts,
    }


def parent_artifacts(trial: pathlib.Path) -> dict:
    """The submitted optimizer and its loss curve -- the node a next attempt modifies.

    AIDE's operators act on a parent node's source; the summary describes history, it does
    not replace the artifact. Without this the next iteration must reimplement from prose.

    The curve is returned at BOTH densities from a single parse of each log, because
    the two have different consumers and must not be conflated: `parent_curve` is
    thinned to ~MAX_CURVE_POINTS for the agent-facing history markdown, while
    `full_curve` is every parsed point and is what `agentloop.reward` scores. A caller
    that scores the thinned curve makes the display budget load-bearing on the reward.

    Keys are ABSENT rather than None when nothing is found, so a caller can splat the
    result into a row and distinguish "no artifact" from "artifact was empty". An empty
    dict is a valid return. `full_curve` is the one key a splatting caller should drop
    first -- see `read_trial`.
    """
    out = {}
    srcs = sorted(trial.glob("artifacts/**/optimizer.py"), key=lambda p: len(p.parts))
    if srcs:
        # Shallowest wins: the submission sits at submission/optimizer.py while working
        # copies and checkpoints accumulate deeper under the same tree.
        text = srcs[0].read_text(errors="replace")
        out["parent_source"] = text[:MAX_PARENT_SOURCE_CHARS]
        out["parent_source_lines"] = text.count("\n") + 1
        out["parent_source_truncated"] = len(text) > MAX_PARENT_SOURCE_CHARS

    curve, full = {}, {}
    for lg in sorted(trial.glob("artifacts/**/full_seed*.log")):
        pts = [(int(m.group(1)), float(m.group(2))) for m in
               VAL_LOSS_RE.finditer(lg.read_text(errors="replace"))]
        if pts:
            seed = lg.stem.replace("full_seed", "")
            full[seed] = pts
            step = max(1, len(pts) // MAX_CURVE_POINTS)
            thinned = pts[::step]
            # The final point is the one the verifier grades on; thinning must never
            # drop it, so it is appended when the stride happens to skip it.
            if pts[-1] not in thinned:
                thinned.append(pts[-1])
            curve[seed] = thinned
    if curve:
        out["parent_curve"] = curve
        out["full_curve"] = full
    return out


def _budget_used(res: dict, trial: pathlib.Path) -> float | None:
    """Fraction of the agent's wall-clock budget the trial consumed.

    `classify` uses this to tell an agent that abandoned its run (budget left on the
    table, no harness exception) from a harness failure, so a None here is meaningful:
    it means the question cannot be answered, not that the budget was unspent.
    """
    try:
        s0 = datetime.fromisoformat(res["started_at"].replace("Z", "+00:00"))
        s1 = datetime.fromisoformat(res["finished_at"].replace("Z", "+00:00"))
    except Exception:
        return None
    budget = DEFAULT_BUDGET_SECS
    try:
        c = json.loads((trial / "config.json").read_text())
        budget = float(c.get("agent", {}).get("timeout_sec") or budget)
    except Exception:
        pass
    return (s1 - s0).total_seconds() / budget if budget else None


def agent_findings(trial: pathlib.Path, cap: int = MAX_FINDINGS_CHARS) -> str:
    """The agent's own closing account, not a truncation of arbitrary text.

    Walks the trajectory backwards for the last substantive assistant message, which is
    where claude-code states what it did and why. harbor writes trajectory.json at trial
    end, so it is absent mid-run and present once the trial completes.
    """
    tj = trial / "agent" / "trajectory.json"
    if not tj.is_file():
        return ""
    try:
        steps = json.loads(tj.read_text()).get("steps") or []
    except Exception:
        return ""
    for step in reversed(steps):
        msg = step.get("message")
        if isinstance(msg, str) and len(msg.strip()) > 200:
            return msg.strip()[:cap]
    return ""


MTIME_SLACK_SECS = 5.0


def _trial_candidates(job_dir: pathlib.Path) -> list[tuple[pathlib.Path, float, float]]:
    """Every child holding a result.json, as (dir, ordering_time, arrival_time).

    Each directory is statted inside the try because a job dir is live: harbor, a
    reaper or an operator can unlink a trial between `glob` yielding it and us
    reading it, and an OSError here would abort an otherwise recoverable iteration.
    A directory that vanished is simply not a candidate.

    The two times answer two different questions and must not be conflated. Ordering
    ("which trial is newest") is mtime, the only signal that ranks trials against
    each other. Arrival ("did this land during THIS iteration") is max(mtime, ctime),
    because mtime is content-modification time and is deliberately preserved by copy,
    extract and archive-mode sync, so a directory placed here seconds ago can carry an
    mtime from days ago; ctime is the inode's own change time and userspace cannot
    backdate it. A trial genuinely left over from an earlier run of this job dir has
    BOTH times old, so this weakens nothing about staleness detection -- it only stops
    a freshly delivered trial from being libelled as stale.
    """
    out = []
    for d in job_dir.glob("*/"):
        try:
            if (d / "result.json").is_file():
                st = d.stat()
                out.append((d, st.st_mtime, max(st.st_mtime, st.st_ctime)))
        except OSError:
            continue
    return out


def find_trial_with_staleness(job_dir: pathlib.Path,
                              since: float) -> tuple[pathlib.Path | None, bool]:
    """The newest trial in `job_dir`, plus whether it PREDATES this iteration.

    The freshness cutoff carries `MTIME_SLACK_SECS` of slack for clock skew between
    the launcher and the filesystem. When nothing clears `since`, the newest surviving
    trial is still returned -- it remains diagnostically useful -- but the second
    element is True, and a caller that scores it as this iteration's result is then
    doing so knowingly rather than by accident.
    """
    cands = _trial_candidates(job_dir)
    if not cands:
        return None, False
    fresh = [c for c in cands if c[2] >= since - MTIME_SLACK_SECS]
    pool = fresh or cands
    return max(pool, key=lambda c: c[1])[0], not fresh


def find_trial(job_dir: pathlib.Path, since: float, *,
               allow_stale: bool = False) -> pathlib.Path | None:
    """The trial this job just produced, or None if it produced none.

    A job dir is reused across relaunches of the same `agentic_iterNN` job, so when
    an iteration produces no trial of its own -- the agent crashed on startup, harbor
    never launched -- the newest thing on disk belongs to an EARLIER run. Handing
    that back indistinguishably makes the loop score a previous iteration's reward,
    curve and parent source as this one's, turning a launch failure into a plausible
    success. So by default a stale trial is refused, loudly.

    `allow_stale=True` opts into the old behaviour for callers that want the stale
    trial for diagnosis; `find_trial_with_staleness` returns it with an explicit
    flag. Both stale paths warn, and the two leading positional parameters are
    unchanged so existing `find_trial(job_dir, t0)` callers keep working.
    """
    trial, is_stale = find_trial_with_staleness(job_dir, since)
    if not is_stale:
        return trial
    print(f"WARNING: agentloop.trial_io.find_trial: no trial in {job_dir} is newer than "
          f"since={since:.0f}; the newest one ({trial}) is STALE and belongs to an "
          f"earlier run of this job dir. "
          + (f"Returning it anyway because allow_stale=True -- it must NOT be scored "
             f"as this iteration's result." if allow_stale else
             f"Returning None: this iteration produced no trial."),
          file=sys.stderr)
    return trial if allow_stale else None


def task_budget_secs(task_dir: pathlib.Path) -> float | None:
    """`[agent].timeout_sec` from a task bundle, or None if it cannot be read."""
    try:
        import tomllib
        with (pathlib.Path(task_dir) / "task.toml").open("rb") as f:
            return float(tomllib.load(f)["agent"]["timeout_sec"])
    except Exception:
        return None


def task_budget_hours(task_dir: pathlib.Path) -> str:
    """The budget as prose for the agent-facing history header.

    Falls back to deferring to the instruction rather than inventing a number, because
    quoting a wrong cap teaches the next iteration to plan against a budget it does
    not have.
    """
    secs = task_budget_secs(task_dir)
    if secs is None:
        return "as stated in the task instruction"
    return f"{secs / 3600:g} hours of wall clock"
