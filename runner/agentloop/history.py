"""Render prior iterations into the markdown block injected into the next prompt.

Ported from track3-pipeline/tools/refine.py (`FORBIDDEN`, `scrub`, `render_history`,
`_restate_budget`, `render_parent`). This is the only module in the loop whose output
is read by the agent, so every rule here is a containment rule, not a formatting one.

Four of them are load-bearing:

* THE SCRUB FIREWALL (ENGRAM's projection-as-a-function). The per-iteration prose is
  written by an LLM summariser reading a trajectory that may quote grader internals.
  Forwarding that verbatim hands the agent the rubric it is being measured against, and
  an agent that can see the rubric optimises the rubric. `FORBIDDEN` is therefore applied
  mechanically to every forwarded block, and redactions are printed to stderr so a leak
  attempt is visible in the run log rather than silent.

* `not graded` IS NOT `0.0000`. A row for a run that never reached the verifier still
  carries reward 0.0. Printed as a number it reads as "your update rule scored zero" and
  teaches the next attempt to abandon an approach that was never actually tested. Only
  `graded_pass`/`graded_miss` -- outcomes where the verifier really produced a score --
  may show a figure.

* EVERY UNGRADED OUTCOME MUST STEER. `classify` returns `unknown` for any verifier reason
  string it does not recognise, and that used to render as outcome `unknown`, reward cell
  `not graded`, and no guidance at all -- the agent was shown a 0.0 with no account of it
  and no instruction, and repeated the failure. Three consecutive unscored iterations in
  the live pipeline came through this hole. `unknown` therefore gets its own block, and
  joins the telemetry-binding set, since an unreadable verdict most often means the
  telemetry never bound.

* THE PARENT GUARD (AIDE's node-based operators). Source is carried forward only for an
  attempt the verifier scored above zero. A scored parent is a proven artifact worth
  mutating; an unscored one is an unvalidated guess, and reproducing 294 lines of it
  anchors the next attempt onto code nothing supports. The summary still carries its
  rule and hyperparameters either way.

DIVERGENCES FROM THE SOURCE (both because this harness is multi-task and has no module
-global TASK directory, and because it has a synthetic backend the original did not):

* `_restate_budget` takes `budget_hours` as a parameter instead of reading a global.
* `render_history` marks fabricated rows: when any row came from a metric-fabricating
  backend, `marking.synthetic_banner` is placed at the TOP of the history, above the
  facts table, where it cannot be mistaken for a real training result. It sits in the
  header chunk, so the length trim can never drop it.
"""
from __future__ import annotations

import re
import sys

from agentloop import summariser
from agentloop.marking import synthetic_banner

MAX_HISTORY_ITERS = 8
MAX_FINDINGS_CHARS = 1200
MAX_HISTORY_CHARS = 30000
MAX_PARENT_SOURCE_CHARS = 18000
MAX_CURVE_POINTS = 12

# Used when the caller cannot resolve the task budget. Deferring to the instruction beats
# inventing a number, because quoting a wrong cap teaches the agent to plan against a
# budget it does not have.
DEFAULT_BUDGET_HOURS = "as stated in the task instruction"

SCORED_OUTCOMES = ("graded_pass", "graded_miss")

FORBIDDEN = (
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


def scrub(text: str) -> tuple[str, list[str]]:
    """Strip grader-internal vocabulary before anything reaches the agent.

    The summariser reads a trajectory that may quote rubric or checker text, so the
    projection is filtered mechanically rather than by trusting the prompt.
    """
    hits = []
    for pat in FORBIDDEN:
        found = re.findall(pat, text, re.IGNORECASE)
        if found:
            hits.extend({f if isinstance(f, str) else str(f) for f in found})
            text = re.sub(pat, "[redacted]", text, flags=re.IGNORECASE)
    return text, sorted(set(hits))


def _restate_budget(text: str, budget_hours: str) -> str:
    """Rewrite stale hour figures in forwarded prose to the cap now in force.

    A summary quotes the budget its own iteration had. Forwarding that verbatim told
    iterations 2 and 3 about a 5-hour cap they never operated under.
    """
    hours = budget_hours.split()[0]
    return re.sub(r"\b\d+(?:\.\d+)?\s*-?\s*hour\b", f"{hours}-hour", text)


def render_history(rows: list[dict], total_iterations: int | None = None,
                   budget_hours: str | None = None) -> str:
    if not rows:
        return ""
    shown = rows[-MAX_HISTORY_ITERS:]
    budget_hours = budget_hours or DEFAULT_BUDGET_HOURS
    n = shown[-1]["iteration"] + 1
    prior = "once" if len(rows) == 1 else f"{len(rows)} times"
    if total_iterations:
        remaining = total_iterations - n
        left = ("this is your last attempt" if remaining <= 0 else
                f"{remaining} further attempt{'s' if remaining > 1 else ''} will follow")
    else:
        left = "further attempts may follow"
    out = [f"# Attempt {n} — your previous attempts at this task", ""]

    # Above everything, including the facts table: a fabricated number that reads as a
    # real one is worse than no history at all.
    banner = synthetic_banner(shown)
    if banner:
        out += [banner, ""]

    out += [
        f"You have attempted this task {prior}. This is attempt {n}; {left}.",
        "",
        "`reward`, `graded_step` and `seeds` below are recorded by the verifier and are not",
        "open to dispute. `outcome` is a classification applied afterwards, and the prose under",
        "each iteration is your own summarised account, written at the time of that attempt.",
        "",
        f"CURRENT budget is {budget_hours}. Earlier accounts may quote a different figure or a",
        "rule that has since changed; where they disagree with the task instruction you have",
        "now, the instruction wins.",
        "",
        "| iter | reward | graded_step | outcome (classified) | seeds |",
        "|---|---|---|---|---|",
    ]
    for r in shown:
        scored = r["outcome"] in SCORED_OUTCOMES
        reward_cell = f"{r['reward']:.4f}" if scored else "not graded"
        out.append(
            f"| {r['iteration']} | {reward_cell} | {r.get('graded_step') or '-'} "
            f"| {r['outcome']} | {r.get('n_seeds', 0)} |"
        )
    out.append("")
    for r in shown:
        block = summariser.render(r.get("summary") or {}) or r.get("findings", "")
        if block:
            block = _restate_budget(block, budget_hours)
            block, redacted = scrub(block)
            if redacted:
                print(f"  [scrub] iteration {r['iteration']}: redacted {redacted}",
                      file=sys.stderr)
            out += [f"## Iteration {r['iteration']} — what you tried", "", block, ""]

    best = max(shown, key=lambda r: r["reward"])
    out += ["## Now attempt the task again", ""]
    if best["reward"] > 0:
        out += [
            f"Your best attempt is iteration {best['iteration']} at reward {best['reward']:.4f}"
            f" (graded_step {best.get('graded_step')}). Its source is reproduced below.",
            "EXPLOIT it -- start from that submission and improve on it -- or, if you judge the",
            "approach exhausted, replace it with a materially different update rule.",
            # DIVERGENCE: the source wraps between "Beat" and the figure. Same prose to
            # the agent, but the target stays one greppable token for the tests.
            f"Beat {best['reward']:.4f}. Submit a graded 2-seed run either way.",
            "",
        ]
    elif any(r["outcome"] in SCORED_OUTCOMES for r in shown):
        out += [
            "No attempt has scored above 0.0, so there is nothing yet worth exploiting.",
            "EXPLORE: try a materially different update rule, or fix what the graded attempts",
            "above got wrong. A 2-seed run that finishes and reconciles is worth more than a",
            "better idea that never gets graded.",
            "",
        ]
    else:
        out += [
            "Nothing above has been graded, so there is no evidence for or against any of it.",
            "Your first job is to produce ONE completed, reconciled 2-seed run and get a real",
            "number on the board. Carrying a previous approach through to a graded result is a",
            "legitimate choice; so is replacing it. What is not useful is another unfinished",
            "run, which teaches nothing.",
            "",
        ]

    if shown[-1]["outcome"] == "agent_abandoned_run":
        out += [
            "## Your last attempt was never graded -- read this carefully",
            "",
            "Your optimizer has no score. Not a bad score: no score. You launched the graded",
            "run, posted a status update saying it was still training, and ended your turn.",
            "The trial ends when you stop producing foreground work, so the container was torn",
            "down mid-training. The verifier then found only a stale telemetry chain left over",
            "from an earlier short plumbing run and rejected it, which is where the recorded",
            "0.0 came from. It measures nothing about your update rule.",
            "",
            "Two consequences. First, the harness logged no error, no timeout and no",
            "environment defect, and most of the budget went unused, so this was your call to",
            "make, not something done to you. Second, your approach is UNTESTED rather than",
            "refuted -- treat the numbers below as probe evidence, not as a verdict.",
            "",
        ]

    if shown[-1]["outcome"] == "unknown":
        out += [
            "## Your last attempt returned no recognisable verdict -- read this carefully",
            "",
            "The run ended without producing a result this loop could read, so it was",
            "recorded as `unknown` and carries 0.0. That 0.0 is bookkeeping, not a",
            "measurement: your update rule was never scored, so this is NOT evidence that",
            "your approach failed, and abandoning it on that basis would be a mistake.",
            "",
            "Usually the run never got as far as being scored -- it was cut short, or its",
            "telemetry did not bind to the optimizer present at the end (see below). Do not",
            "redesign around this. Produce ONE completed, reconciled 2-seed run and get a",
            "real number on the board; only then is there anything to judge the idea by.",
            "",
        ]

    if any(r["outcome"] in ("agent_abandoned_run", "harness_incomplete", "gate_fail",
                            "unknown")
           for r in shown):
        out += [
            "## Telemetry binding",
            "",
            "The telemetry chain is append-only across every run in this container, and every",
            "record must hash-match the optimizer.py present at grading time. If you probe with",
            "one optimizer and then edit it, the probe's records will not match and the reward",
            "is 0.0. Send probe telemetry elsewhere with TRACK3_TELEMETRY_DIR and keep",
            "/telemetry for the graded run only.",
            "",
        ]

    narrative = "\n".join(out)
    if len(narrative) > MAX_HISTORY_CHARS:
        # The header, banner and facts table are the part that must never be dropped: they
        # are the verifier's settled numbers. Only the agent's own retold prose is
        # sacrificed, oldest first, since the newest account is the most actionable.
        header, *blocks = narrative.split("\n## ")
        per_iteration = [b for b in blocks if b.startswith("Iteration ")]
        mechanics = [b for b in blocks if not b.startswith("Iteration ")]
        joined = lambda: "\n## ".join([header, *per_iteration, *mechanics])
        while per_iteration and len(joined()) > MAX_HISTORY_CHARS:
            del per_iteration[0]
        narrative = joined()

    return narrative + render_parent(max(shown, key=lambda r: (r["reward"], r["iteration"])))


def render_parent(carried: dict) -> str:
    """Carry the source only for an attempt the verifier actually graded above zero.

    A scored parent is a proven artifact and reimplementing it from prose loses detail, so
    exploitation wants the bytes. An unscored parent is an unvalidated guess: handing back
    294 lines invites editing a line nothing supports, and anchors away from the new update
    rule the task asks for. The summary still carries its rule and hyperparameters.
    """
    if not carried.get("parent_source") or carried.get("reward", 0) <= 0:
        return ""
    out = ["", f"## Your previous submission (iteration {carried['iteration']})", ""]
    if carried.get("parent_curve"):
        out += ["Validation curve it produced:", "", "```"]
        for seed, pts in sorted(carried["parent_curve"].items()):
            out.append(f"seed{seed}  " + "  ".join(f"{st}:{vl:.5f}" for st, vl in pts))
        out += ["```", ""]
    note = " (truncated)" if carried.get("parent_source_truncated") else ""
    out += [f"This is the exact `submission/optimizer.py` you submitted{note}. Modify it,",
            "or replace it, but do not reimplement it blind.", "",
            "```python", carried["parent_source"].rstrip(), "```", ""]
    return "\n".join(out)
