"""The concern registry and its MECE proof.

Adapted from kaiju/verification/coverage.py:40 `registry_violations`, which is the best idea in
that codebase: the partition is PROVED by executable code rather than asserted in a docstring.

Extended here in three ways kaiju does not cover:

  * step coverage — every concern binds to a declared truth step, every step is covered, and a
    step covered only by concerns that may not run is rejected (checklist B1-B3);
  * final-outcome coverage — the step marked final must carry BOTH halves (B4), because an
    outcome judged only in bytes or only in prose is half-judged;
  * implementation binding — a declared deterministic concern must resolve to a real predicate
    and vice versa (A6, A7), so the registry cannot describe an instrument that does not exist.

`violations()` returning empty IS the proof. `test_mece.py` plants each violation class and
requires the prover to catch it, because a prover that cannot fail proves nothing.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .schemas import Concern, Dimension, Layer, Owner
from .truth import TruthSpec


class Registry:
    """A set of concerns, a truth spec, and the predicates that implement them."""

    def __init__(self, concerns: Iterable[Concern], truth: TruthSpec,
                 implemented: Iterable[str] | None = None):
        self.concerns: list[Concern] = list(concerns)
        self.truth = truth
        self.implemented: set[str] | None = None if implemented is None else set(implemented)

    # ---- views ------------------------------------------------------------------ #
    @property
    def deterministic(self) -> list[Concern]:
        return [c for c in self.concerns if c.owner is Owner.DETERMINISTIC]

    @property
    def rubric(self) -> list[Concern]:
        return [c for c in self.concerns if c.owner is Owner.RUBRIC]

    def by_step(self, owner: Owner | None = None) -> dict[str, list[Concern]]:
        out: dict[str, list[Concern]] = defaultdict(list)
        for c in self.concerns:
            if owner is None or c.owner is owner:
                out[c.truth_step].append(c)
        return dict(out)

    def coverage_map(self) -> dict[str, dict[str, list[str]]]:
        det = self.by_step(Owner.DETERMINISTIC)
        rub = self.by_step(Owner.RUBRIC)
        return {s.id: {"deterministic": sorted(c.id for c in det.get(s.id, [])),
                       "rubric": sorted(c.id for c in rub.get(s.id, []))}
                for s in self.truth.steps}

    # ---- the proof --------------------------------------------------------------- #
    def violations(self) -> list[str]:
        """Return every way this registry fails to be MECE. Empty list means sound."""
        v: list[str] = []
        ids = [c.id for c in self.concerns]
        det_ids = {c.id for c in self.deterministic}
        rub_ids = {c.id for c in self.rubric}

        # -- mutual exclusivity ---------------------------------------------------- #
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            v.append(f"ME: duplicate concern ids: {dupes}")
        both = sorted(det_ids & rub_ids)
        if both:
            v.append(f"ME: ids owned by both halves: {both}")
        for c in self.rubric:
            if c.layer is not Layer.JUDGMENT:
                v.append(f"ME: rubric concern {c.id} sits in layer {c.layer.name}; a judged "
                         "criterion in a decidable layer belongs in the deterministic half")
        for c in self.deterministic:
            if c.layer is Layer.JUDGMENT:
                v.append(f"ME: deterministic concern {c.id} sits in layer JUDGMENT; if it needs "
                         "judgement it cannot be asserted by pytest")

        # -- implementation binding ------------------------------------------------ #
        if self.implemented is not None:
            missing = sorted(det_ids - self.implemented)
            if missing:
                v.append(f"binding: declared but unimplemented deterministic concerns: {missing}")
            orphan = sorted(self.implemented - det_ids)
            if orphan:
                v.append(f"binding: implemented predicates with no declared concern: {orphan}")

        # -- collective exhaustiveness over the golden path ------------------------ #
        steps = set(self.truth.step_ids)
        for c in self.concerns:
            if c.truth_step not in steps:
                v.append(f"CE: concern {c.id} binds to unknown truth step {c.truth_step!r}")

        det_by_step = self.by_step(Owner.DETERMINISTIC)
        rub_by_step = self.by_step(Owner.RUBRIC)
        for s in self.truth.steps:
            here_det = det_by_step.get(s.id, [])
            here_rub = rub_by_step.get(s.id, [])
            if not here_det and not here_rub:
                v.append(f"CE: truth step {s.id} has no coverage of either kind")
                continue
            if not here_det:
                v.append(f"CE: truth step {s.id} has no deterministic coverage")
            elif not any(c.always_measurable for c in here_det):
                # A step whose only deterministic concerns may report NOT_MEASURED has, in
                # production, no deterministic signal at all -- its whole weight falls through to
                # the judged half. That is how a narrative comes to stand in for evidence.
                v.append(f"CE: truth step {s.id} is covered only by concerns that may not run; "
                         "its weight would fall through to the judged half in production")
            if s.final_outcome and not here_rub:
                v.append(f"CE: final-outcome step {s.id} has no judged coverage; an outcome "
                         "judged only in bytes is half-judged")
            if s.final_outcome and not here_det:
                v.append(f"CE: final-outcome step {s.id} has no deterministic coverage")

        # -- weights ---------------------------------------------------------------- #
        for c in self.concerns:
            if c.weight <= 0:
                v.append(f"weight: concern {c.id} has non-positive weight {c.weight}")
        for s in self.truth.steps:
            if s.weight <= 0:
                v.append(f"weight: truth step {s.id} has non-positive weight {s.weight}")

        # -- dimension sanity -------------------------------------------------------- #
        for c in self.concerns:
            if c.gating and c.dimension is not Dimension.LEGITIMACY:
                v.append(f"dimension: {c.id} is gating but its dimension is "
                         f"{c.dimension.value}; only legitimacy may gate, or a low score and a "
                         "disqualification become the same event")
            if c.dimension is Dimension.LEGITIMACY and not c.gating:
                v.append(f"dimension: {c.id} is legitimacy but not gating; a legitimacy concern "
                         "that does not gate is scored nowhere and therefore inert")
        return v

    def assert_sound(self) -> "Registry":
        v = self.violations()
        if v:
            raise MECEError("registry is not MECE:\n  - " + "\n  - ".join(v))
        return self

    def proof(self) -> dict:
        det, rub = self.deterministic, self.rubric
        return {
            "deterministic_concerns": len(det),
            "rubric_concerns": len(rub),
            "id_overlap": len({c.id for c in det} & {c.id for c in rub}),
            "truth_steps": len(self.truth.steps),
            "final_outcome_steps": self.truth.final_steps,
            "steps_with_deterministic_coverage": len(self.by_step(Owner.DETERMINISTIC)),
            "steps_with_judged_coverage": len(self.by_step(Owner.RUBRIC)),
            "implemented_predicates": (len(self.implemented)
                                         if self.implemented is not None else None),
            "violations": self.violations(),
        }


class MECEError(AssertionError):
    pass
