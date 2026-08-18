#!/usr/bin/env python3
"""Emit the delivery-format verifier artifacts for one graded minicalc run.

Harbor parses EVERY key of the score file it reads as a number, so nothing published
in `verifier/score.json` may be a string. That single constraint is why this file
exists: grade.py's machine-readable `reason` cannot survive into the only artifact
harbor reads, so it travels as the integer `reason_code` below and the prose stays in
`verifier/grade-stdout.md`, which this module also completes with the full record.

Everything here is derived from files the run already produced -- the graded record in
grade-stdout.md, the pytest transcript in test-stdout.md, the per-seed logs -- and
nothing is carried over from a previous score.json.

DIVERGENCES FROM THE bia/track3nov EMITTER THIS MIRRORS, ALL DELIBERATE:

* No `_loss_from_numeric` carry-over. That exists in track3nov because its FineWeb
  seed logs are too large to ship, so a repackaged bundle has to read its own prior
  score.json back. minicalc's logs are a few thousand lines and `task.toml` collects
  `/workspace/submission/logs` as an artifact, so they are always present. Dropping
  the fallback removes the only path by which a stale measurement could outlive a
  re-grade.
* Seed logs are found by RECURSIVE glob. track3nov globs `artifacts/full_seed*.log`;
  harbor collects minicalc's to `artifacts/workspace/submission/logs/full_seed*.log`,
  and in the container they are under `/workspace/submission` while the run dir is
  `/logs`. The recursive glob plus the optional seed-source argument covers both.
* Absent evidence is an ABSENT key, never a zero, and that extends to the pytest
  block: `_pytest_counts` returns nothing at all when the transcript carries no
  summary line, because a `pytests_passed: 0` beside a `pytests_executed: 0` reads as
  a suite that ran and asserted nothing. pytest is not installed in bia/minicalc:v1,
  so that is the normal case here, not an edge one.
* No rubric handling beyond absence. The LLM judge (`runner/agentloop/judge.py`) is
  opt-in and off, so no `rubric_verdicts.json` is ever written and the `rubrics_*`
  keys never appear.

`verifier/score.md` is written by test.sh, not here, so the bare score exists even
when this module is skipped -- test.sh calls it with `|| true` precisely so a defect
in these review aids can never fail a legitimate grade.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

SEED_LOG_GLOB = "**/full_seed*.log"
VAL_LOSS_RE = re.compile(r"step:(\d+)/\d+\s+val_loss:([\d.]+)")
COUNT_WORDS = ("passed", "failed", "skipped")


def _pytest_counts(log: str) -> dict:
    """Counts from a pytest summary line, or `{}` when the transcript has none.

    An empty dict means "no pytest evidence", which is what a skip notice or a
    "command not found" leaves behind, and it is what keeps the `pytests_*` keys out
    of score.json entirely instead of publishing a row of zeros that looks like a
    suite which ran and asserted nothing.
    """
    tail = log.strip().splitlines()[-1] if log.strip() else ""
    found = {w: re.search(rf"(\d+) {w}", tail) for w in COUNT_WORDS}
    if not any(found.values()):
        return {}
    counts = {w: int(m.group(1)) if m else 0 for w, m in found.items()}
    counts["executed"] = counts["passed"] + counts["failed"]
    counts["failed_tests"] = sorted(set(
        re.findall(r"^FAILED \S+::(\w+)", log, re.M)))
    return counts


def _seed_curves(root: pathlib.Path) -> dict:
    curves = {}
    for lg in sorted(root.glob(SEED_LOG_GLOB)):
        try:
            text = lg.read_text(errors="replace")
        except OSError:
            continue
        pts = {int(m.group(1)): float(m.group(2))
               for m in VAL_LOSS_RE.finditer(text)}
        if pts:
            curves[lg.stem.replace("full_", "")] = pts
    return curves


def _loss(root: pathlib.Path, graded_step):
    """Mean and per-seed val_loss at the graded step, or None when nothing parsed.

    Falls back to the last step EVERY seed logged when the graded step is absent from
    the intersection, so a run that missed the target still reports a comparable
    number rather than nothing.
    """
    curves = _seed_curves(root)
    if not curves:
        return None
    common = sorted(set.intersection(*(set(c) for c in curves.values())))
    if not common:
        return None
    at = graded_step if graded_step in common else max(common)
    return {"steps": at,
            "at_graded_step": round(sum(c[at] for c in curves.values()) / len(curves), 6),
            "per_seed": {k: round(v[at], 6) for k, v in curves.items()}}


# Codes group by the gate that produced the reason rather than running sequentially, so
# a new reason joins its family without renumbering codes already recorded in shipped
# runs. No entry is a prefix of another, so the first match is the only match.
#
# This table is minicalc's OWN vocabulary and its numbers are NOT comparable with
# bia/track3nov's: minicalc runs no HMAC telemetry chain, no submission binding, no
# frozen-recipe check and no novelty corpus, so codes for those families would
# advertise gates this verifier does not have.
#
#   0   graded pass
#   1x  the verifier itself produced no result file
#   2x  graded miss -- the run was measured and did not earn reward
#   3x  no telemetry at all
#   4x  insufficient seed coverage
#
# The two 2x codes are kept distinct because reason_code is the only thing that
# survives into score.json, and "no seed ever reached the target" and "every seed
# reached it, too late" are the difference between a wrong algorithm and an untuned
# one -- the first thing a reviewer needs and cannot recover from the score.
REASON_CODES = (
    ("graded_step=", 0),
    ("verifier_produced_no_reward", 10),
    ("verifier_produced_no_score", 10),
    ("no_step_clears_target_loss", 20),
    ("no_step_clears_baseline_reached_target_at_step_", 21),
    ("telemetry_absent", 30),
    ("need_at_least_", 40),
)
REASON_CODE_MISSING = -1
REASON_CODE_UNRECOGNISED = 99


def _reason_code(reason) -> int:
    if not reason:
        return REASON_CODE_MISSING
    for prefix, code in REASON_CODES:
        if reason.startswith(prefix):
            return code
    return REASON_CODE_UNRECOGNISED


def _graded_record(grade_stdout: str) -> dict:
    for line in grade_stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return {}


def build(run: pathlib.Path, seed_source: pathlib.Path | None = None) -> tuple[dict, dict]:
    v = run / "verifier"
    grade = (v / "grade-stdout.md").read_text() if (v / "grade-stdout.md").is_file() else ""
    rec = _graded_record(grade)
    # `score` first, `reward` second: grade.py emits `reward`, the track3nov lineage
    # emits `score`, and a record from either grades the same way here.
    score = float(rec.get("score", rec.get("reward", 0.0)))
    reason = rec.get("reason", "")
    step = int(reason.split("=")[1]) if reason.startswith("graded_step=") else None

    pt = _pytest_counts((v / "test-stdout.md").read_text()) if (v / "test-stdout.md").is_file() else {}
    verd = {}
    if (run / "rubric_verdicts.json").is_file():
        verd = json.loads((run / "rubric_verdicts.json").read_text()).get("verdicts", {})
    rb_pass = sum(1 for x in verd.values() if x.get("pass"))

    pt_frac = (pt.get("passed", 0) / pt["executed"]) if pt.get("executed") else None
    rb_frac = (rb_pass / len(verd)) if verd else None
    composite = score * (pt_frac if pt_frac is not None else 1.0) \
        * (rb_frac if rb_frac is not None else 1.0)

    loss = _loss(seed_source or run, step)

    numeric = {"score": score, "graded_score": score, "composite": round(composite, 6),
               "reason_code": _reason_code(reason)}
    if step is not None:
        numeric["graded_step"] = step
    for k in ("passed", "failed", "skipped", "executed"):
        if k in pt:
            numeric[f"pytests_{k}"] = pt[k]
    if pt_frac is not None:
        numeric["pytests_fraction"] = round(pt_frac, 6)
    if verd:
        numeric.update(rubrics_passed=rb_pass, rubrics_total=len(verd),
                       rubrics_fraction=round(rb_frac, 6))
    if loss:
        numeric["loss_at_graded_step"] = loss["at_graded_step"]
        numeric["loss_steps"] = loss["steps"]
        for s, val in loss["per_seed"].items():
            numeric[f"loss_per_seed_{s}"] = val

    full = dict(rec)
    full.update({
        "score": score,
        "composite": round(composite, 6),
        "reason_code": numeric["reason_code"],
        "formula": "graded_score * (pytests_passed/executed) * (rubrics_passed/total), "
                   "each factor omitted when that evidence is absent",
        "pytests": pt or None,
        "rubrics": ({"passed": rb_pass, "total": len(verd),
                     "failed": [k for k, x in verd.items() if not x.get("pass")]}
                    if verd else None),
        "loss": loss,
        "note": ("reward is the graded value the verifier computed and is authoritative. "
                 "composite is a review aid bounded by it and can never exceed it. "
                 "score.json carries only numeric keys because harbor parses them all as "
                 "numbers; this record carries the rest. A null pytests or rubrics block "
                 "means that evidence was never produced, not that it scored zero."),
    })
    return numeric, full


FULL_RECORD_MARKER = "--- full record ---"


def write_full_record(verifier: pathlib.Path, full: dict) -> None:
    """Append the full record to grade-stdout.md, below grade.py's own output.

    Splitting on the marker and keeping only the head makes this idempotent: the
    graded line grade.py printed is preserved and a re-run replaces the record instead
    of stacking a second copy underneath the first. `build` reads the FIRST line
    starting with `{`, so the graded line stays the record it parses on every pass.
    """
    path = verifier / "grade-stdout.md"
    head = path.read_text().split(FULL_RECORD_MARKER)[0].rstrip() if path.is_file() else ""
    body = json.dumps(full, indent=2, sort_keys=True)
    path.write_text(f"{head}\n\n{FULL_RECORD_MARKER}\n{body}\n" if head
                    else f"{FULL_RECORD_MARKER}\n{body}\n")


def main() -> int:
    run = pathlib.Path(sys.argv[1])
    seed = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
    numeric, full = build(run, seed)
    (run / "verifier" / "score.json").write_text(
        json.dumps(numeric, indent=1, sort_keys=True) + "\n")
    write_full_record(run / "verifier", full)
    print(f"{run.name}: score={numeric['score']} composite={numeric['composite']} "
          f"reason_code={numeric['reason_code']} numeric_keys={len(numeric)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
