"""Validation-line parser for track-3 run logs.

Robustness is a graded property here, not a convenience. Real record logs interleave
torchrun rank warnings with validation lines, carry timestamps, and vary in indentation,
so the committed parser must recover the same step-to-loss mapping under all of those
perturbations. The VERIFIER-ROBUSTNESS class replays recorded agent logs through this
exact function under every bound perturbation and requires an identical score.
"""
from __future__ import annotations

import re

# Anchored on the frozen trainer's own emission format. Leading content is permitted so
# that interleaved rank prefixes and timestamps do not hide a line, but the step and loss
# fields themselves are matched exactly.
VAL_LINE = re.compile(r"step:\s*(\d+)\s*/\s*\d+\s+val_loss:\s*([0-9]*\.?[0-9]+)")


def parse_log(text: str) -> dict:
    """Return {step: val_loss}. Later occurrences of a step overwrite earlier ones."""
    out = {}
    for m in VAL_LINE.finditer(text):
        out[int(m.group(1))] = float(m.group(2))
    return out


def parse_file(path: str) -> dict:
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse_log(f.read())
