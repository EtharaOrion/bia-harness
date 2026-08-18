"""The consolidated reward — one scalar, with its components left visible beside it.

Kaiju refuses to fuse. It emits a gate, a graded_score and an honesty signal as three separate
fields (kaiju/verification/schemas.py:66-105) and `counts_toward_score` explicitly excludes
legitimacy and honesty (:51-55). That refusal is principled: the three answer different questions
and averaging them destroys information.

But a benchmark needs one number per run, and "the consumer must fuse them" means every consumer
invents a different fusion. So bia-verifier defines it ONCE, in one auditable place, and keeps
every component beside it so the fusion can be inspected, disputed, or replaced without re-deriving
anything.

THE DEFINITION

    consolidated = 0.0                                    if the gate is QUARANTINE
    consolidated = sum_over_steps(weight * disposition)   otherwise

    disposition = deterministic_fraction * judged_multiplier
    judged_multiplier = 1 - MAX_JUDGED_PENALTY * failed_judged_weight_fraction

FOUR PROPERTIES, EACH DELIBERATE

1. A gate failure is a ZERO, not a deduction. A run that fabricated its evidence has not "scored
   poorly"; it has disqualified itself, and a fused number that let it place above an honest
   partial run would be worse than no number.

2. The judged half may only DAMPEN. The multiplier is at most 1.0, so prose can never lift a score
   its artifacts did not earn. This mirrors the veto-only convention kaiju uses for its rubric
   layer (rubric_layer.py:89-95, gating=False, honesty dimension).

3. The multiplier is driven by the VERDICT, not the judge's numeric confidence. A criterion is met
   or it is not. Multiplying by a confidence score has two failure modes I measured directly: the
   same passing account scored 0.85 on one call and 0.9 on the next, so the instrument was not
   reproducible; and full marks became unreachable even for a run that met every criterion, since
   no judge returns exactly 1.0 on everything. The numeric score is retained and reported.

4. Honesty does not enter the arithmetic. It is a lens, reported alongside. An honestly incomplete
   run must not be taxed for saying so, and a deceptive one is caught by the gate, not by a
   fractional penalty that lets it trade honesty for points.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .schemas import CheckStatus, Dimension, Gate, StepScore

# How much of a step the judged half may remove. 0.5 means a worthless account costs half of what
# the artifacts proved and never more. A policy constant, declared here rather than inlined so it
# is auditable in one place; see CHECKLIST.md gap G-c.
MAX_JUDGED_PENALTY = 0.5


@dataclass
class RewardBreakdown:
    consolidated_reward: float | None
    gate: str
    verification_complete: bool
    process_score: float | None
    deterministic_ceiling: float
    judged_penalty_applied: float
    honesty_signal: str
    task_reward: float | None = None
    task_reward_note: str = ""
    steps: list[dict] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def judged_multiplier(failed_weight_fraction: float | None) -> float:
    """[1 - MAX_JUDGED_PENALTY, 1.0]. 1.0 when the judged half returned no verdict.

    Unavailable is not the same as failing: a judge that could not run must not silently tax a run
    whose artifacts are sound.
    """
    if failed_weight_fraction is None:
        return 1.0
    f = max(0.0, min(1.0, float(failed_weight_fraction)))
    return max(1.0 - MAX_JUDGED_PENALTY, min(1.0, 1.0 - MAX_JUDGED_PENALTY * f))


def deterministic_fraction(statuses: list[CheckStatus]) -> tuple[float, bool]:
    """Fraction of DECIDED deterministic checks that passed, plus an all-unmeasured flag.

    NOT_MEASURED is excluded from the denominator: it is neither a pass nor a failure, and
    counting it either way is a lie about what the instrument did. When every check for a step is
    unmeasured the fraction is 0.0 and the flag is set, so the caller can tell "nothing was
    proved" from "everything failed".
    """
    decided = [s for s in statuses if s in (CheckStatus.PASS, CheckStatus.FAIL)]
    if not decided:
        return 0.0, bool(statuses)
    return sum(1 for s in decided if s is CheckStatus.PASS) / len(decided), False


def consolidate(report, *, task_reward: float | None = None) -> RewardBreakdown:
    """Fuse a finalized VerificationReport into one scalar without hiding its parts."""
    ceiling = round(sum(s.weight * s.deterministic_fraction for s in report.steps), 6)
    total = round(sum(s.weight * s.disposition for s in report.steps), 6)

    quarantined = report.gate is Gate.QUARANTINE
    complete = bool(report.meta.get("verification_complete"))
    if not complete:
        consolidated = None
    elif quarantined:
        consolidated = 0.0
    else:
        consolidated = max(0.0, min(1.0, total))

    penalty = round(ceiling - total, 6) if ceiling >= total else 0.0
    honesty = (report.meta.get("honesty") or {}).get("signal", "unevaluated")

    note = ("consolidated_reward fuses the gate and the per-step process score. It is 0.0 on "
            "QUARANTINE because a disqualified run has not scored poorly, it has disqualified "
            "itself. The judged half may only dampen, never raise. Honesty is reported beside "
             "the number and never enters it.")
    if not complete:
        note += (" Verification is incomplete, so consolidated_reward is null and this run is "
                 "not rankable.")
    if quarantined:
        note += (" This run is QUARANTINED: "
                 + ", ".join(r.concern_id for r in report.gating_failures) + ".")

    return RewardBreakdown(
        consolidated_reward=(round(consolidated, 6) if consolidated is not None else None),
        gate=report.gate.value,
        verification_complete=complete,
        process_score=report.process_score,
        deterministic_ceiling=ceiling,
        judged_penalty_applied=penalty,
        honesty_signal=honesty,
        task_reward=task_reward,
        task_reward_note=("The task's own reward, carried beside and NEVER blended: it answers a "
                          "different question (did the artifact meet the task's bar) than the "
                          "instrument (was the golden path walked and evidenced)."),
        steps=[s.to_dict() for s in report.steps],
        note=note,
    )


def score_steps(truth, results, contract_by_step) -> list[StepScore]:
    """Turn per-concern results into the weighted per-step curve.

    `contract_by_step` maps step id -> {"deterministic": [concern ids], "rubric": [concern ids]}.
    """
    by_id = {r.concern_id: r for r in results}
    rows: list[StepScore] = []
    cumulative = 0.0
    for step in truth.steps:
        cov = contract_by_step.get(step.id, {})
        det_ids = cov.get("deterministic", [])
        rub_ids = cov.get("rubric", [])

        det_statuses = [by_id[c].status for c in det_ids if c in by_id]
        frac, all_nm = deterministic_fraction(det_statuses)

        verdicted = failed = 0.0
        for cid in rub_ids:
            r = by_id.get(cid)
            if r is None or r.status not in (CheckStatus.PASS, CheckStatus.FAIL):
                continue
            verdicted += r.weight
            if r.status is CheckStatus.FAIL:
                failed += r.weight
        failed_fraction = (failed / verdicted) if verdicted > 0 else None
        mult = judged_multiplier(failed_fraction)

        # A judgement can dampen artifact-backed credit but can never replace missing artifacts.
        if all_nm:
            disposition = 0.0
        else:
            disposition = frac * mult
            assert disposition <= frac + 1e-12, "judged half raised a step"

        cumulative += step.weight * disposition
        rows.append(StepScore(
            truth_step=step.id,
            weight=round(step.weight, 6),
            deterministic_fraction=round(frac, 6),
            judged_multiplier=round(mult, 6),
            disposition=round(disposition, 6),
            cumulative=round(cumulative, 6),
            deterministic_detail={c: by_id[c].status.value for c in det_ids if c in by_id},
            judged_detail={c: by_id[c].status.value for c in rub_ids if c in by_id},
            all_not_measured=all_nm,
        ))
    return rows
