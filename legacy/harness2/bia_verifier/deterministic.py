"""The deterministic half: predicates, and the proof that each one can fail.

A predicate answers one DETERMINISTIC concern from the evidence of one run. It is a plain callable
so a dataset supplies its own without subclassing anything, and so the core never learns what any
particular harness looks like.

WHAT THIS ADDS OVER A BARE DICT OF CALLABLES
--------------------------------------------
`prune_vacuous` (after kaiju/verification/predicates.py:91-101). Every predicate declares a
NEGATIVE FIXTURE -- evidence on which it MUST fail. A predicate whose negative fixture passes
cannot fail on anything, so it is dropped rather than shipped. A check that cannot go red is not a
check; it is decoration that makes a suite look thorough.

`stub_floor` (after kaiju/verification/tiers.py:75). The inverse of the oracle guarantee: the
frozen set must FAIL on a stub. Without it, a suite where everything passes would satisfy
"the oracle scores 100%" perfectly and mean nothing at all.

`two_sided_mutation` (after kaiju/verification/mutation.py:129-133). One-sided mutation testing
rewards a verifier that rejects everything. Two-sided requires that BREAKING mutants are killed
AND PRESERVING mutants are spared, so over-strictness is a failure too.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

from .schemas import CheckStatus


@dataclass
class PredicateOutcome:
    status: CheckStatus
    summary: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is CheckStatus.PASS


@dataclass
class Predicate:
    """One deterministic check, bound to a concern, with proof it can fail."""
    concern_id: str
    fn: Callable[[Any], PredicateOutcome]
    negative_fixture: Any = None
    description: str = ""

    def __call__(self, evidence: Any) -> PredicateOutcome:
        return self.fn(evidence)

    @property
    def has_negative_fixture(self) -> bool:
        return self.negative_fixture is not None


@dataclass
class VacuityReport:
    kept: list[str] = field(default_factory=list)
    vacuous: list[str] = field(default_factory=list)
    unproven: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.vacuous

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept": sorted(self.kept),
            "vacuous_dropped": sorted(self.vacuous),
            "unproven_no_negative_fixture": sorted(self.unproven),
            "ok": self.ok,
            "note": ("A predicate is VACUOUS when its own negative fixture passes: it cannot fail "
                     "on anything, so it measures nothing. UNPROVEN predicates ship but carry no "
                     "demonstration that they can go red -- they are listed so their status is "
                     "never mistaken for proven."),
        }


def prune_vacuous(predicates: Iterable[Predicate]) -> tuple[dict[str, Predicate], VacuityReport]:
    """Drop every predicate that cannot fail on its own negative fixture."""
    rep = VacuityReport()
    kept: dict[str, Predicate] = {}
    for p in predicates:
        if not p.has_negative_fixture:
            rep.unproven.append(p.concern_id)
            kept[p.concern_id] = p
            continue
        try:
            outcome = p(p.negative_fixture)
        except Exception:  # noqa: BLE001
            # A predicate that raises on its negative fixture DID refuse it. Raising is a poor
            # way to say no, but it is not a vacuous pass, so the predicate survives.
            kept[p.concern_id] = p
            rep.kept.append(p.concern_id)
            continue
        if outcome.passed:
            rep.vacuous.append(p.concern_id)
            continue
        kept[p.concern_id] = p
        rep.kept.append(p.concern_id)
    return kept, rep


def stub_floor(results: dict[str, CheckStatus]) -> bool:
    """True when the frozen set FAILS on a stub, which is what makes a green run meaningful."""
    if not results:
        return False
    return any(s is CheckStatus.FAIL for s in results.values())


@dataclass
class MutationReport:
    killed_breaking: list[str] = field(default_factory=list)
    survived_breaking: list[str] = field(default_factory=list)
    spared_preserving: list[str] = field(default_factory=list)
    killed_preserving: list[str] = field(default_factory=list)

    @property
    def sound(self) -> bool:
        """Every breaking mutant killed AND every preserving mutant spared."""
        return not self.survived_breaking and not self.killed_preserving

    @property
    def score(self) -> float:
        total = len(self.killed_breaking) + len(self.survived_breaking)
        return (len(self.killed_breaking) / total) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(sound=self.sound, score=round(self.score, 6),
                 note=("Two-sided. A surviving BREAKING mutant means the suite missed a real "
                       "defect; a killed PRESERVING mutant means it is too strict and punishes "
                       "an equivalent implementation. One-sided mutation testing would reward a "
                       "verifier that rejects everything."))
        return d


def two_sided_mutation(breaking: dict[str, Any], preserving: dict[str, Any],
                       run: Callable[[Any], bool]) -> MutationReport:
    """`run(mutant) -> True` means the verifier ACCEPTED the mutant."""
    rep = MutationReport()
    for name, mutant in sorted(breaking.items()):
        (rep.survived_breaking if run(mutant) else rep.killed_breaking).append(name)
    for name, mutant in sorted(preserving.items()):
        (rep.spared_preserving if run(mutant) else rep.killed_preserving).append(name)
    return rep
