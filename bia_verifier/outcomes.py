"""Loader the generated pytest suite reads. Kept tiny and prose-free on purpose.

The generated tests import only `state` and `why` from here, so the suite carries ids and the
implied relation and nothing else. A missing outcome is FAIL, never a pass: a check that never ran
is a defect in the verifier, and defaulting it to anything else is how a suite silently stops
testing.
"""
from __future__ import annotations

import json
import os

OUTCOMES_ENV = "BIA_VERIFIER_OUTCOMES"
DEFAULT = "/tmp/bia_verifier_outcomes.json"


def path() -> str:
    return os.environ.get(OUTCOMES_ENV, DEFAULT)


def load() -> dict:
    with open(path()) as f:
        return json.load(f)


def state(concern_id: str) -> str:
    rec = load().get(concern_id)
    if rec is None:
        return "fail"
    return rec.get("status", "fail")


def why(concern_id: str) -> str:
    rec = load().get(concern_id) or {}
    return (f"{concern_id}: status={rec.get('status')} summary={rec.get('summary')!r} "
            f"evidence={json.dumps(rec.get('evidence') or {}, sort_keys=True)[:300]}")
