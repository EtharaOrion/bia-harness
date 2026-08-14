"""Anchoring — the mechanism that makes the oracle score 100% by construction.

Adopted from kaiju/verification/rubric_anchor.py:127-149 and pytest_exec.py:117-145, which is the
most valuable idea in that codebase.

THE PROBLEM IT SOLVES. If criteria are hand-authored, nothing stops one from being unsatisfiable
by the reference solution. I hit this concretely in a predecessor pipeline: a criterion asked why
an optimizer's rule DIFFERS from the published lineage, while the declared oracle was an avowed
recombination that differs from nothing. The oracle could not pass its own instrument, and the
defect was invisible until an independent judge was pointed at it.

THE RULE. A criterion is admitted to the frozen set only if:

    the GOLDEN passes it        AND        the STUB fails it

Failing the first makes it unsound — it asks for something the reference solution does not do.
Failing the second makes it non-discriminating — it is satisfied by a submission that did nothing,
so it measures nothing. Both classes are dropped, with the reason recorded.

THE GUARANTEE, STATED HONESTLY. An oracle re-scored on the anchored set passes 100% of it, because
anything the golden failed was removed. That is a construction, not a hope. Two limits, both
reported in every run rather than buried:

  * it covers ANCHORABLE criteria only — a criterion about the shape of an agent's search cannot
    be anchored against a bare solution, since a solution has no trajectory (kaiju has the same
    limit at verifier_loop.py:132);
  * it is only as stable as the evaluator — anchoring against a stochastic judge admits criteria
    that a later call might score differently, which is why the anchor records the backend it used
    and why verdicts, not numeric scores, drive the result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

DROP_UNSOUND = "golden_failed_it"
DROP_NON_DISCRIMINATING = "stub_also_passed_it"
KEEP = "kept"
UNANCHORABLE = "unanchorable"


@dataclass
class AnchorVerdict:
    criterion_id: str
    disposition: str
    golden_passed: bool | None = None
    stub_passed: bool | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnchorReport:
    kept: list[str] = field(default_factory=list)
    dropped: list[AnchorVerdict] = field(default_factory=list)
    unanchorable: list[str] = field(default_factory=list)
    backend: str = "unknown"
    min_required: int = 1

    @property
    def ok(self) -> bool:
        return len(self.kept) >= self.min_required

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept": sorted(self.kept),
            "dropped": [d.to_dict() for d in self.dropped],
            "unanchorable": sorted(self.unanchorable),
            "backend": self.backend,
            "min_required": self.min_required,
            "ok": self.ok,
            "note": ("A criterion is kept only if the golden passes it AND the stub fails it. "
                     "The oracle therefore scores 100% of `kept` by construction. `unanchorable` "
                     "criteria are NOT covered by that guarantee and are listed so the guarantee "
                     "is never read as broader than it is."),
        }


def anchor(criterion_ids: Iterable[str],
           evaluate_golden: Callable[[str], bool | None],
           evaluate_stub: Callable[[str], bool | None],
           *,
           anchorable: Callable[[str], bool] | None = None,
           backend: str = "unknown",
           min_required: int = 1) -> AnchorReport:
    """Partition criteria into kept / dropped / unanchorable.

    `evaluate_*` return True (passed), False (failed) or None (could not evaluate). None is NOT
    treated as a pass in either direction: a criterion the golden could not be evaluated against
    is unanchorable, not admitted.
    """
    rep = AnchorReport(backend=backend, min_required=min_required)
    for cid in criterion_ids:
        if anchorable is not None and not anchorable(cid):
            rep.unanchorable.append(cid)
            continue
        g = evaluate_golden(cid)
        if g is None:
            rep.unanchorable.append(cid)
            continue
        if not g:
            rep.dropped.append(AnchorVerdict(cid, DROP_UNSOUND, g, None,
                                             "the reference solution does not satisfy it, so it "
                                             "asks for something the golden path does not do"))
            continue
        s = evaluate_stub(cid)
        if s is None:
            rep.unanchorable.append(cid)
            continue
        if s:
            rep.dropped.append(AnchorVerdict(cid, DROP_NON_DISCRIMINATING, g, s,
                                             "a submission that did nothing also satisfies it, "
                                             "so it separates nothing"))
            continue
        rep.kept.append(cid)
    return rep


def stub_floor_ok(stub_results: dict[str, bool]) -> bool:
    """The inverse of the oracle guarantee (kaiju/verification/tiers.py:75).

    The frozen set must FAIL on a stub. Without this, a suite that passes everything would satisfy
    the oracle requirement perfectly and mean nothing at all.
    """
    return bool(stub_results) and not all(stub_results.values())
