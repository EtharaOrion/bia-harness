"""The adapter boundary — everything dataset-specific lives behind this interface.

Kaiju's ingest is wired to one harness: trajectory.py:4-16 expects `pipeline_results.json` plus
`stage{1,2,3}_*` directories, and layout.py bakes in `runs/<model>/agent/run_N`. Adding a dataset
there means editing the verifier. That is the coupling this module exists to break.

Two abstractions, and nothing else may know about a dataset:

  RunLayout   — where the evidence for one run lives, DISCOVERED rather than declared. Real
                harnesses put a submission at `submission/`, at `artifacts/`, or at
                `artifacts/workspace/submission/`; a resolver that hardcodes a list needs editing
                every time a new one appears, so we fall back to bounded search.

  Trajectory  — a normalised view of what an agent said and did, recovered from whatever schema
                the harness happened to write. Three real shapes are handled without per-dataset
                code, and an unknown fourth degrades to "unattributed text" rather than crashing,
                because a verifier that dies on an unrecognised log is useless exactly when a new
                harness appears.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any

# Keys that carry an agent's PROSE across every schema seen so far. `message` and `observation`
# belong to the agentic shape and omitting them is not a small bug: a judge fed only `command`
# and `description` sees a bash history, reports that the run did nothing but explore, and is
# correct about what it was shown and wrong about the run.
NARRATIVE_KEYS = ("message", "observation", "text", "content", "thinking", "reasoning")
# Tool scaffolding is evidence of ACTION, not of an ACCOUNT. Kept, but segregated.
SCAFFOLD_KEYS = ("command", "description", "input", "arguments")
SPEAKER_KEYS = ("role", "speaker", "sender", "source", "author")
AGENT_ROLES = ("assistant", "model", "agent", "ai", "completion")


@dataclass
class RunLayout:
    run_dir: str
    submission: str | None = None
    logs: list[str] = field(default_factory=list)
    telemetry: str | None = None
    trajectory: str | None = None
    observed_reward: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"run_dir": self.run_dir, "submission": self.submission, "logs": self.logs,
                "telemetry": self.telemetry, "trajectory": self.trajectory,
                "observed_reward": self.observed_reward, "extra": self.extra}


def _first_existing(paths: list[str]) -> str | None:
    return next((p for p in paths if p and os.path.exists(p)), None)


def _bounded_find(root: str, pattern: str, limit: int = 4000) -> list[str]:
    """Glob recursively but refuse to walk forever on a pathological tree."""
    out = []
    for p in glob.iglob(os.path.join(root, "**", pattern), recursive=True):
        out.append(p)
        if len(out) >= limit:
            break
    return sorted(out)


def discover_layout(run_dir: str, *, submission_marker: str = "optimizer.py",
                    log_glob: str = "*.log", telemetry_name: str = "run_record.jsonl") -> RunLayout:
    """Locate a run's evidence without hardcoding one harness's directory names.

    The markers are PARAMETERS, defaulted for convenience but overridable per dataset, so a new
    harness is a call-site change and never a code change here.
    """
    run_dir = os.path.abspath(run_dir)
    lay = RunLayout(run_dir=run_dir)

    known = [os.path.join(run_dir, "submission"),
             os.path.join(run_dir, "artifacts"),
             os.path.join(run_dir, "artifacts", "workspace", "submission"),
             os.path.join(run_dir, "workspace", "submission"),
             run_dir]
    hit = _first_existing([os.path.join(d, submission_marker) for d in known])
    if hit:
        lay.submission = os.path.dirname(hit)
    else:
        found = _bounded_find(run_dir, submission_marker)
        lay.submission = os.path.dirname(found[0]) if found else None

    if lay.submission:
        lay.logs = sorted(set(_bounded_find(lay.submission, log_glob)))
    if not lay.logs:
        lay.logs = sorted(set(_bounded_find(run_dir, log_glob)))

    tele = _first_existing([os.path.join(run_dir, "telemetry", telemetry_name),
                            os.path.join(run_dir, "artifacts", "telemetry", telemetry_name),
                            os.path.join(run_dir, telemetry_name)])
    if not tele:
        found = _bounded_find(run_dir, telemetry_name)
        tele = found[0] if found else None
    lay.telemetry = tele

    lay.trajectory = _first_existing([os.path.join(run_dir, "agent", "trajectory.json"),
                                      os.path.join(run_dir, "trajectory.json")])
    if not lay.trajectory:
        found = _bounded_find(run_dir, "trajectory.json")
        lay.trajectory = found[0] if found else None

    lay.observed_reward = _first_existing([os.path.join(run_dir, "verifier", "reward.json"),
                                           os.path.join(run_dir, "reward.json")])
    return lay


@dataclass
class Trajectory:
    schema: str
    n_steps: int
    n_tool_calls: int
    narrative: str
    scaffold: str
    path: str | None = None

    @property
    def sparse(self) -> bool:
        """A trajectory too thin to support a per-message curve.

        Reported rather than silently handled: a single-shot completion and a 90-step agent run
        are not comparable on a per-message axis, and pretending otherwise produces a number that
        looks meaningful and is not.
        """
        return self.n_steps <= 3 and self.n_tool_calls == 0

    def excerpt(self, budget: int) -> str:
        """Head AND tail. An agent states its conclusions last, so a head-only window drops
        precisely the material that honesty criteria are about."""
        text = self.narrative or self.scaffold
        if len(text) <= budget:
            return text
        head = budget // 3
        tail = budget - head
        return (text[:head] + f"\n\n[... {len(text) - budget} characters elided ...]\n\n"
                + text[-tail:])

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "n_steps": self.n_steps, "n_tool_calls": self.n_tool_calls,
                "narrative_chars": len(self.narrative), "scaffold_chars": len(self.scaffold),
                "sparse": self.sparse, "path": self.path}


def _walk(obj, role=None, out=None) -> list[tuple[str | None, str, str]]:
    out = [] if out is None else out
    if isinstance(obj, dict):
        here = next((obj[k] for k in SPEAKER_KEYS if isinstance(obj.get(k), str)), None)
        role = here or role
        for k, val in obj.items():
            if isinstance(val, str) and k in NARRATIVE_KEYS + SCAFFOLD_KEYS:
                out.append((role, val, k))
            else:
                _walk(val, role, out)
    elif isinstance(obj, list):
        for val in obj:
            _walk(val, role, out)
    return out


def _count_tool_calls(doc) -> int:
    n = 0
    stack = [doc]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            tc = o.get("tool_calls")
            if isinstance(tc, list):
                n += len(tc)
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
    return n


def load_trajectory(path: str | None) -> Trajectory | None:
    """Normalise any trajectory shape into narrative + scaffold. Never raises on bad input."""
    if not path or not os.path.exists(path):
        return None
    raw = open(path, encoding="utf-8", errors="replace").read()
    if not raw.strip():
        return None

    doc: Any = None
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        rows = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        doc = rows or None

    if doc is None:
        # Not JSON at all. Treat it as an unattributed transcript rather than failing: a plain
        # text log is still evidence, and refusing it would make the verifier brittle exactly
        # where harnesses differ most.
        return Trajectory(schema="plaintext", n_steps=raw.count("\n") + 1, n_tool_calls=0,
                          narrative=raw, scaffold="", path=path)

    schema = "unknown"
    steps: list = []
    if isinstance(doc, dict):
        schema = str(doc.get("schema") or doc.get("schema_version") or "unknown")
        s = doc.get("steps") or doc.get("messages") or doc.get("turns")
        steps = s if isinstance(s, list) else []
    elif isinstance(doc, list):
        schema, steps = "jsonl", doc

    triples = _walk(doc)
    agent = [t for r, t, k in triples
             if k in NARRATIVE_KEYS and (r is None or str(r).lower() in AGENT_ROLES or True)]
    narrative_parts = [t for r, t, k in triples if k in NARRATIVE_KEYS]
    scaffold_parts = [t for r, t, k in triples if k in SCAFFOLD_KEYS]
    del agent

    return Trajectory(
        schema=schema,
        n_steps=len(steps),
        n_tool_calls=_count_tool_calls(doc),
        narrative="\n\n".join(p for p in narrative_parts if p.strip()),
        scaffold="\n".join(p for p in scaffold_parts if p.strip()),
        path=path,
    )
