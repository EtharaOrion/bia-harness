"""The golden path as STRUCTURE, not prose.

Kaiju's TRUTH.md is seven prose headings split on `##` (kaiju/verification/truth.py:20,90) with no
step objects and no criterion binding. That is enough to tell a reader what the task is; it is not
enough to say how far along the path a run travelled, because nothing connects a criterion to a
stage of the work.

Here a TruthSpec is an ordered list of steps, each carrying a weight, and every concern in the
registry must name the step it evidences. That single binding is what makes:

  * collective exhaustiveness checkable (does every step have coverage?),
  * a continuous score meaningful (how much of the path was walked and evidenced?),
  * and a final-outcome step distinguishable from the process that leads to it.

A TruthSpec is DATA. It is loaded from the dataset, never hardcoded here — this module knows what
a golden path IS, not what any particular golden path SAYS.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TruthStep:
    id: str
    action: str
    establishes: str
    weight: float
    final_outcome: bool = False
    note: str = ""


@dataclass
class TruthSpec:
    task: str
    steps: list[TruthStep] = field(default_factory=list)
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def step_ids(self) -> list[str]:
        return [s.id for s in self.steps]

    @property
    def final_steps(self) -> list[str]:
        return [s.id for s in self.steps if s.final_outcome]

    def step(self, sid: str) -> TruthStep | None:
        return next((s for s in self.steps if s.id == sid), None)

    def weight_of(self, sid: str) -> float:
        s = self.step(sid)
        return s.weight if s else 0.0

    def normalised(self) -> "TruthSpec":
        """Return a copy whose step weights sum to exactly 1.0.

        Normalising rather than rejecting lets a dataset author express weights in whatever units
        are natural (percentages, points, relative importance) without the pipeline caring.
        """
        total = sum(s.weight for s in self.steps)
        if total <= 0:
            n = len(self.steps) or 1
            steps = [TruthStep(s.id, s.action, s.establishes, 1.0 / n, s.final_outcome, s.note)
                     for s in self.steps]
        else:
            steps = [TruthStep(s.id, s.action, s.establishes, s.weight / total, s.final_outcome,
                               s.note) for s in self.steps]
        return TruthSpec(task=self.task, steps=steps, source=self.source, meta=dict(self.meta))

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "source": self.source,
                "steps": [asdict(s) for s in self.steps], "meta": self.meta}


class TruthError(ValueError):
    pass


def load(path: str) -> TruthSpec:
    """Load a TruthSpec from JSON or YAML. The format is the dataset's business, not ours."""
    if not os.path.exists(path):
        raise TruthError(f"truth spec not found: {path}")
    raw = open(path, encoding="utf-8").read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        doc = yaml.safe_load(raw)
    else:
        doc = json.loads(raw)
    return from_dict(doc, source=path)


def from_dict(doc: dict, source: str = "") -> TruthSpec:
    if not isinstance(doc, dict):
        raise TruthError("truth spec must be a mapping")
    steps_raw = doc.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise TruthError("truth spec declares no steps; a golden path with no steps cannot be "
                         "walked, scored, or bound to")
    steps: list[TruthStep] = []
    seen: set[str] = set()
    for i, s in enumerate(steps_raw):
        if not isinstance(s, dict) or not s.get("id"):
            raise TruthError(f"step {i} has no id")
        sid = str(s["id"])
        if sid in seen:
            raise TruthError(f"duplicate step id {sid!r}")
        seen.add(sid)
        steps.append(TruthStep(
            id=sid,
            action=str(s.get("action", "")),
            establishes=str(s.get("establishes", "")),
            weight=float(s.get("weight", 1.0)),
            final_outcome=bool(s.get("final_outcome", False)),
            note=str(s.get("note", "")),
        ))
    if not any(s.final_outcome for s in steps):
        # Refusing to guess is deliberate. If nothing is marked final there is no outcome to
        # judge, and silently treating the last step as final would hide an authoring mistake
        # behind a plausible default.
        raise TruthError("no step is marked final_outcome; the golden path has no endpoint to "
                         "judge, so a run could never be scored on its result")
    return TruthSpec(task=str(doc.get("task", "")), steps=steps, source=source,
                     meta=dict(doc.get("meta") or {}))


def render_markdown(spec: TruthSpec, coverage: dict[str, dict[str, list[str]]] | None = None) -> str:
    """Render the golden path, optionally annotated with which concerns cover each step.

    The annotation is the point: a TRUTH document that does not say what enforces each step is a
    description, and descriptions drift from the instrument without anyone noticing.
    """
    L = [f"# Golden path — {spec.task}", "",
         "GENERATED. Derived from the truth spec; do not hand-edit.", ""]
    for s in spec.steps:
        L.append(f"## {s.id}. {s.action}")
        L.append("")
        L.append(f"Establishes: {s.establishes}")
        L.append(f"Weight: {s.weight:g}{'  (FINAL OUTCOME)' if s.final_outcome else ''}")
        if coverage:
            cov = coverage.get(s.id, {})
            L.append(f"- deterministic: {', '.join(cov.get('deterministic', [])) or 'none'}")
            L.append(f"- judged: {', '.join(cov.get('rubric', [])) or 'none'}")
        if s.note:
            L.append(f"- note: {s.note}")
        L.append("")
    return "\n".join(L)
