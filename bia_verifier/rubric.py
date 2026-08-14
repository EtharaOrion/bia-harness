"""The judged half: criteria a machine cannot decide from bytes.

A RubricItem is the RUBRIC-owned counterpart of a deterministic predicate. It carries the
criterion a judge must apply, the guidance that disambiguates it, and the concern id that binds it
into the MECE registry -- so a rubric item can never float free of the partition.

Two rules inherited from kaiju's rubric layer, both learned the hard way there:

  * `sanitize()` strips references to the answer key (kaiju/verification/rubric.py:105). A
    criterion that says "as described in TRUTH.md" leaks the golden path into the judge prompt and
    turns judging into recall.
  * An inconclusive judgement is NOT a failure (kaiju/verification/rubric_layer.py:32-40). A judge
    that could not run must not manufacture a verdict against the run.

And one rule that is ours: `anchorable` marks whether a criterion can be evaluated against a bare
reference solution. A criterion about the SHAPE OF A SEARCH cannot be -- a solution has no
trajectory -- so it is excluded from the anchoring guarantee rather than silently weakening it.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .schemas import CheckStatus

_ANSWER_KEY_REFS = re.compile(
    r"\b(truth\.md|truth doc(?:ument)?|the golden (?:path|diff|solution)|answer key|"
    r"reference solution)\b", re.I)

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_INCONCLUSIVE = "inconclusive"
DECIDED_VERDICTS = (VERDICT_PASS, VERDICT_FAIL)


class RubricError(ValueError):
    pass


@dataclass(frozen=True)
class RubricItem:
    concern_id: str
    criterion: str
    guidance: str = ""
    anchorable: bool = False
    weight: float = 1.0

    def sanitized(self) -> "RubricItem":
        """Remove answer-key references so judging cannot degrade into recall."""
        return RubricItem(
            concern_id=self.concern_id,
            criterion=_ANSWER_KEY_REFS.sub("the stated requirements", self.criterion).strip(),
            guidance=_ANSWER_KEY_REFS.sub("the stated requirements", self.guidance).strip(),
            anchorable=self.anchorable,
            weight=self.weight,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    concern_id: str
    verdict: str
    score: float | None = None
    rationale: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def decided(self) -> bool:
        return self.verdict in DECIDED_VERDICTS

    def to_status(self) -> CheckStatus:
        """Map a verdict to a check status.

        `inconclusive` becomes NOT_MEASURED, never FAIL. Fabricating a failure because the judge
        could not answer would make an infrastructure outage indistinguishable from a bad run.
        """
        if self.verdict == VERDICT_PASS:
            return CheckStatus.PASS
        if self.verdict == VERDICT_FAIL:
            return CheckStatus.FAIL
        return CheckStatus.NOT_MEASURED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Rubric:
    items: list[RubricItem] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return [i.concern_id for i in self.items]

    def anchorable(self) -> list[RubricItem]:
        return [i for i in self.items if i.anchorable]

    def sanitized(self) -> "Rubric":
        return Rubric([i.sanitized() for i in self.items])

    def restricted_to(self, keep: Iterable[str]) -> "Rubric":
        """Freeze the rubric to the anchored set. Anything the golden failed is gone."""
        k = set(keep)
        return Rubric([i for i in self.items if i.concern_id in k])

    def to_dict(self) -> dict[str, Any]:
        return {"items": [i.to_dict() for i in self.items]}

    @staticmethod
    def from_dict(doc: dict) -> "Rubric":
        if not isinstance(doc, dict):
            raise RubricError("rubric must be a JSON/YAML object")
        items = doc.get("items", [])
        if not isinstance(items, list):
            raise RubricError("rubric.items must be a list")
        rubric = Rubric([RubricItem(
            concern_id=i["concern_id"], criterion=i["criterion"],
            guidance=i.get("guidance", ""), anchorable=bool(i.get("anchorable", False)),
            weight=float(i.get("weight", 1.0))) for i in items])
        rubric.validate()
        return rubric

    @staticmethod
    def load(path: str) -> "Rubric":
        import json

        raw = open(path, encoding="utf-8").read()
        if path.endswith((".yaml", ".yml")):
            import yaml
            return Rubric.from_dict(yaml.safe_load(raw))
        return Rubric.from_dict(json.loads(raw))

    def validate(self, expected_ids: Iterable[str] | None = None) -> "Rubric":
        ids = self.ids
        duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
        if duplicates:
            raise RubricError(f"duplicate rubric concern ids: {duplicates}")
        for item in self.items:
            if not item.concern_id.strip():
                raise RubricError("rubric item has an empty concern_id")
            if not item.criterion.strip():
                raise RubricError(f"rubric item {item.concern_id!r} has an empty criterion")
            if item.weight <= 0:
                raise RubricError(
                    f"rubric item {item.concern_id!r} has non-positive weight {item.weight}")
        if expected_ids is not None:
            expected = set(expected_ids)
            actual = set(ids)
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            if missing or extra:
                raise RubricError(
                    f"rubric/spec id mismatch: missing={missing}, extra={extra}")
        return self


def fill_omissions(items: Iterable[RubricItem], verdicts: dict[str, Verdict]) -> dict[str, Verdict]:
    """Any criterion the judge did not answer becomes a FAIL.

    Conservative in the opposite direction from `inconclusive`, and deliberately so: silence about
    a specific criterion is not the same as a judge that could not run. If omission were treated as
    unmeasured, a judge could pass an item simply by never mentioning it
    (kaiju/verification/judge.py:116-135).
    """
    out = dict(verdicts)
    for it in items:
        if it.concern_id not in out:
            out[it.concern_id] = Verdict(
                it.concern_id, VERDICT_FAIL, 0.0,
                "the judge returned no verdict for this criterion; silence cannot earn a pass",
                {"omitted": True})
    return out
