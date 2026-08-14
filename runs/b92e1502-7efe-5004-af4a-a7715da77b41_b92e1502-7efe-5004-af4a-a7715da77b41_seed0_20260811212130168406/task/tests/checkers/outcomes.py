"""Loader for the per-rubric compiled outcomes that grade.py emits.

test_output.py asserts against these booleans and carries no criterion prose and no
reference text, which invariant 24 requires.
"""
from __future__ import annotations

import json
import os

OUTCOMES = os.environ.get("TRACK3_OUTCOMES", "/tmp/track3_outcomes.json")


def load() -> dict:
    with open(OUTCOMES) as f:
        return json.load(f)
