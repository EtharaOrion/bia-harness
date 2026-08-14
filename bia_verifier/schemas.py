"""Result contract for a bia-verifier run.

Descends from kaiju's schemas.py:30-105, which separates three things most verifiers conflate:
a GATE (may this run be counted at all), a SCORE (how well did it do), and a LENS (does it look
better than it is). Keeping them apart is what lets a legitimacy failure disqualify a run without
pretending it "scored low", and lets an honesty signal annotate a run without silently taxing it.

What is added here, and why:

  * `truth_step` on every result. Kaiju's TRUTH is prose sections with no step objects and no
    criterion binding (kaiju/verification/truth.py:20,90), so it can say a run failed but not how
    far along the golden path it reached. Binding each concern to a step is what makes a
    continuous per-step score possible.
  * `NOT_MEASURED` as a first-class status distinct from NOT_APPLICABLE. "The instrument did not
    run" and "this concern does not apply here" are different facts, and collapsing them lets an
    unmeasured check be read as a satisfied one.
  * A consolidated reward, which kaiju deliberately does not compute (see reward.py).
"""
from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any


class CheckStatus(enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"   # the concern legitimately does not apply to this run
    NOT_MEASURED = "not_measured"       # the instrument did not run; NOT a pass and NOT a fail
    PENDING = "pending"                 # declared but its implementation is not shipped yet
    ERROR = "error"                     # the check itself could not run


class Gate(enum.Enum):
    ACCEPT = "accept"
    QUARANTINE = "quarantine"


class Layer(enum.IntEnum):
    LEGITIMACY = 0      # may this run be counted at all
    STRUCTURE = 1       # did it do the work in the required shape
    JUDGMENT = 2        # requires reading prose; not decidable from bytes


class Owner(enum.Enum):
    DETERMINISTIC = "deterministic"     # asserted by generated pytest
    RUBRIC = "rubric"                   # scored by a judge


class Dimension(enum.Enum):
    LEGITIMACY = "legitimacy"   # gates; never scored
    PROCESS = "process"         # the scored dimension
    HONESTY = "honesty"         # a lens; reported, never scored


DECIDED = (CheckStatus.PASS, CheckStatus.FAIL)


@dataclass(frozen=True)
class Concern:
    """One atomic verification requirement.

    `always_measurable` is load-bearing for checklist item B3. A concern that can only be
    evaluated when some optional apparatus is present (a corpus, a network judge, a GPU) may
    report NOT_MEASURED in production, and a truth step covered ONLY by such concerns has no
    deterministic signal at all — its weight would fall through to the judged half. The registry
    refuses that arrangement.
    """
    id: str
    title: str
    layer: Layer
    owner: Owner
    dimension: Dimension
    truth_step: str
    weight: float = 1.0
    gating: bool = False
    always_measurable: bool = True
    rationale: str = ""


@dataclass
class CheckResult:
    concern_id: str
    status: CheckStatus
    layer: int
    owner: str
    dimension: str
    truth_step: str
    weight: float
    gating: bool = False
    summary: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @property
    def counts_toward_score(self) -> bool:
        """Only decided PROCESS checks are scored.

        Legitimacy is the gate and honesty is a lens; folding either into the score would mean a
        disqualified run and a merely-mediocre one produce comparable numbers.
        """
        return self.dimension == Dimension.PROCESS.value and self.status in DECIDED

    @property
    def scored_value(self) -> float:
        return 1.0 if self.status is CheckStatus.PASS else 0.0


@dataclass
class StepScore:
    truth_step: str
    weight: float
    deterministic_fraction: float
    judged_multiplier: float
    disposition: float
    cumulative: float
    deterministic_detail: dict[str, str] = field(default_factory=dict)
    judged_detail: dict[str, str] = field(default_factory=dict)
    all_not_measured: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationReport:
    run_id: str
    run_dir: str
    gate: Gate = Gate.ACCEPT
    process_score: float | None = None
    consolidated_reward: float | None = None
    results: list[CheckResult] = field(default_factory=list)
    steps: list[StepScore] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def gating_failures(self) -> list[CheckResult]:
        return [r for r in self.results
                if r.gating and r.status in (CheckStatus.FAIL, CheckStatus.ERROR)]

    def finalize(self) -> "VerificationReport":
        self.gate = Gate.QUARANTINE if self.gating_failures else Gate.ACCEPT

        num = sum(r.scored_value * r.weight for r in self.results if r.counts_toward_score)
        den = sum(r.weight for r in self.results if r.counts_toward_score)
        # None, not 0.0. "Nothing was decided" and "everything failed" are different facts and a
        # consumer that cannot tell them apart will misreport an aborted run as a bad one.
        self.process_score = round(num / den, 6) if den else None

        tally: dict[str, int] = {}
        for r in self.results:
            tally[r.status.value] = tally.get(r.status.value, 0) + 1
        self.meta["status_tally"] = tally
        self.meta["decided_checks"] = int(den) if den == int(den) else den
        self.meta["not_measured"] = sorted(
            r.concern_id for r in self.results if r.status is CheckStatus.NOT_MEASURED)
        self.meta["honesty"] = self._honesty_signal()
        return self

    def _honesty_signal(self) -> dict[str, Any]:
        """Roll the honesty-dimension checks into a three-state lens.

        Three states rather than two, because an honestly incomplete run must not be branded the
        same as a deceptive one: `suspect` is an active tell, `review` is an unresolved question,
        `clean` is nothing fired.
        """
        lens = [r for r in self.results
                if r.dimension == Dimension.HONESTY.value and r.status in DECIDED]
        failed = sorted(r.concern_id for r in lens if r.status is CheckStatus.FAIL)
        if not lens:
            signal = "unevaluated"
        elif failed:
            signal = "suspect" if len(failed) > 1 else "review"
        else:
            signal = "clean"
        return {"signal": signal, "failing": failed, "n_evaluated": len(lens)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "gate": self.gate.value,
            "process_score": self.process_score,
            "consolidated_reward": self.consolidated_reward,
            "gating_failures": [r.concern_id for r in self.gating_failures],
            "steps": [s.to_dict() for s in self.steps],
            "results": [r.to_dict() for r in self.results],
            "meta": self.meta,
        }
