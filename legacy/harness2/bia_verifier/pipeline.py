"""End-to-end: a dataset spec in, a graded report with a consolidated reward out.

The whole point of the adapter boundary is that this module contains no knowledge of any
particular harness. A dataset is a JSON/YAML spec naming its truth steps, its concerns, and the
markers its layout uses; the predicates are supplied by the caller as plain callables. Adding a
dataset never edits this file.

ONE JUDGEMENT PER RUN. Every artifact derives from the same evaluation pass. In a predecessor
pipeline the rubric file and the report file each triggered their own judge call, so the two saved
artifacts disagreed with each other -- one recorded a failed criterion the other had never seen.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .adapters import RunLayout, Trajectory, discover_layout, load_trajectory
from .anchor import AnchorReport
from .registry import Registry
from .reward import RewardBreakdown, consolidate, score_steps
from .schemas import (
    CheckResult, CheckStatus, Concern, Dimension, Layer, Owner, VerificationReport,
)
from .truth import TruthSpec, from_dict as truth_from_dict

# A predicate answers ONE deterministic concern from the evidence of ONE run.
Predicate = Callable[["Evidence"], "PredicateOutcome"]


@dataclass
class PredicateOutcome:
    status: CheckStatus
    summary: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """Everything a predicate or judge may read about one run."""
    run_id: str
    layout: RunLayout
    trajectory: Trajectory | None
    observed_reward: dict | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSpec:
    """A dataset, expressed as data. Nothing here is code."""
    name: str
    truth: TruthSpec
    concerns: list[Concern]
    submission_marker: str = "optimizer.py"
    log_glob: str = "*.log"
    telemetry_name: str = "run_record.jsonl"
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(doc: dict) -> "DatasetSpec":
        truth = truth_from_dict(doc["truth"]).normalised()
        concerns = [
            Concern(
                id=c["id"], title=c.get("title", ""),
                layer=Layer[c["layer"].upper()],
                owner=Owner[c["owner"].upper()],
                dimension=Dimension[c["dimension"].upper()],
                truth_step=c["truth_step"],
                weight=float(c.get("weight", 1.0)),
                gating=bool(c.get("gating", False)),
                always_measurable=bool(c.get("always_measurable", True)),
                rationale=c.get("rationale", ""),
            )
            for c in doc["concerns"]
        ]
        return DatasetSpec(
            name=doc.get("name", ""), truth=truth, concerns=concerns,
            submission_marker=doc.get("submission_marker", "optimizer.py"),
            log_glob=doc.get("log_glob", "*.log"),
            telemetry_name=doc.get("telemetry_name", "run_record.jsonl"),
            meta=dict(doc.get("meta") or {}),
        )

    @staticmethod
    def load(path: str) -> "DatasetSpec":
        raw = open(path, encoding="utf-8").read()
        if path.endswith((".yaml", ".yml")):
            import yaml
            return DatasetSpec.from_dict(yaml.safe_load(raw))
        return DatasetSpec.from_dict(json.loads(raw))


def gather_evidence(run_dir: str, spec: DatasetSpec, run_id: str | None = None) -> Evidence:
    layout = discover_layout(run_dir, submission_marker=spec.submission_marker,
                             log_glob=spec.log_glob, telemetry_name=spec.telemetry_name)
    observed = None
    if layout.observed_reward:
        try:
            observed = json.load(open(layout.observed_reward))
        except (OSError, json.JSONDecodeError):
            observed = None
    return Evidence(run_id=run_id or os.path.basename(os.path.abspath(run_dir)),
                    layout=layout, trajectory=load_trajectory(layout.trajectory),
                    observed_reward=observed)


def grade(spec: DatasetSpec,
          evidence: Evidence,
          predicates: dict[str, Predicate],
           judgements: dict[str, CheckStatus] | None = None,
           judgement_evidence: dict[str, dict[str, Any]] | None = None,
          *,
          anchor_report: AnchorReport | None = None,
          task_reward: float | None = None) -> tuple[VerificationReport, RewardBreakdown]:
    """Evaluate one run. `judgements` are the ALREADY-COMPUTED rubric verdicts."""
    reg = Registry(spec.concerns, spec.truth, set(predicates)).assert_sound()
    judgements = judgements or {}
    judgement_evidence = judgement_evidence or {}
    rubric_ids = {c.id for c in spec.concerns if c.owner is Owner.RUBRIC}
    if set(judgements) != rubric_ids:
        missing = sorted(rubric_ids - set(judgements))
        extra = sorted(set(judgements) - rubric_ids)
        raise ValueError(f"judgement/spec id mismatch: missing={missing}, extra={extra}")
    results: list[CheckResult] = []

    for c in spec.concerns:
        if c.owner is Owner.DETERMINISTIC:
            fn = predicates.get(c.id)
            if fn is None:
                out = PredicateOutcome(CheckStatus.PENDING, "no predicate registered")
            else:
                try:
                    out = fn(evidence)
                except Exception as e:  # noqa: BLE001
                    # A predicate that raises is a VERIFIER defect. It must name itself rather
                    # than dissolve into a generic failure attributed to the run.
                    out = PredicateOutcome(CheckStatus.ERROR, f"{type(e).__name__}: {e}",
                                           {"predicate": c.id})
            status, summary, ev = out.status, out.summary, out.evidence
        else:
            status = judgements.get(c.id, CheckStatus.NOT_MEASURED)
            detail = judgement_evidence.get(c.id, {})
            summary = str(detail.get("rationale", ""))[:400]
            ev = dict(detail)

        results.append(CheckResult(
            concern_id=c.id, status=status, layer=int(c.layer), owner=c.owner.value,
            dimension=c.dimension.value, truth_step=c.truth_step, weight=c.weight,
            gating=c.gating, summary=summary, evidence=ev,
        ))

    report = VerificationReport(run_id=evidence.run_id, run_dir=evidence.layout.run_dir,
                                results=results)
    report.steps = score_steps(spec.truth, results, reg.coverage_map())
    report.finalize()

    report.meta["dataset"] = spec.name
    report.meta["mece_proof"] = reg.proof()
    report.meta["trajectory"] = evidence.trajectory.to_dict() if evidence.trajectory else None
    report.meta["layout"] = evidence.layout.to_dict()
    concerns_by_id = {c.id: c for c in spec.concerns}
    incomplete = sorted(
        r.concern_id for r in results
        if r.status in (CheckStatus.PENDING, CheckStatus.ERROR)
        or (r.status is CheckStatus.NOT_MEASURED
            and (r.owner == Owner.RUBRIC.value
                 or concerns_by_id[r.concern_id].always_measurable))
    )
    report.meta["verification_complete"] = not incomplete
    report.meta["incomplete_checks"] = incomplete
    if anchor_report is not None:
        report.meta["anchoring"] = anchor_report.to_dict()

    breakdown = consolidate(report, task_reward=task_reward)
    report.consolidated_reward = breakdown.consolidated_reward
    return report, breakdown


def write_artifacts(out_dir: str, report: VerificationReport,
                    breakdown: RewardBreakdown,
                    judgements: dict[str, Any] | None = None) -> dict[str, str]:
    """Write every artifact from the SAME evaluation, so they cannot disagree.

    `outcomes.json` is what the generated pytest suite asserts against. It is derived from the
    very results that produced the report, never from a second evaluation pass: in a predecessor
    pipeline the rubric file and the report file each triggered their own judge call, and the two
    saved artifacts recorded different verdicts for the same run.
    """
    os.makedirs(out_dir, exist_ok=True)
    outcomes = {r.concern_id: {"status": r.status.value, "summary": r.summary,
                               "evidence": r.evidence, "truth_step": r.truth_step}
                for r in report.results}
    payloads = {
        "report.json": report.to_dict(),
        "reward.json": breakdown.to_dict(),
        "outcomes.json": outcomes,
    }
    if judgements is not None:
        payloads["rubric.json"] = judgements
    paths = {}
    for name, payload in payloads.items():
        p = os.path.join(out_dir, name)
        with open(p, "w") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
        paths[name] = p
    return paths
