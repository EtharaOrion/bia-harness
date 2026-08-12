#!/usr/bin/env python3
"""Track-3 speedrun grader. Stdlib only.

Parses a training log for `step:S/N val_loss:V` lines, computes step_to_3_28
and derived metrics, emits Harbor-format JSON: {"reward": float, ...metrics}.

Reward mapping:
    reward = clip((3500 - step_to_3_28) / (3500 - 2500), 0, 1) if val <= 3.28
    reward = 0                                                  otherwise

Rationale:
    - 3500 = canonical Muon baseline (rewarded 0)
    - 2500 = ambitious stretch (rewarded 1)
    - Sub-linear at extremes, linear in the interesting band
    - Statsig margin from Track-3 rules is emitted separately at n=1 sample

Usage:
    grader.py <log-file>        -> prints JSON to stdout
    grader.py --write <log> <out.json>  -> writes to file
"""
import json
import math
import re
import sys
from pathlib import Path

TARGET_VAL_LOSS = 3.28
BASELINE_STEPS = 3500   # canonical Muon
STRETCH_STEPS = 2500    # ambitious upper end of reward curve
LOG_LINE_RE = re.compile(r"step:(\d+)/(\d+)\s+val_loss:([\d.]+)(?:\s+train_time:([\d.]+)s)?(?:\s+step_avg:([\d.]+)ms)?")


def parse_log(text: str) -> dict:
    checkpoints = []
    total_steps = None
    train_time_s = None
    for line in text.splitlines():
        m = LOG_LINE_RE.search(line)
        if not m:
            continue
        step, total, val = int(m.group(1)), int(m.group(2)), float(m.group(3))
        total_steps = total
        if m.group(4):
            train_time_s = float(m.group(4))
        checkpoints.append({"step": step, "val_loss": val})
    return {
        "checkpoints": checkpoints,
        "total_steps": total_steps,
        "final_train_time_s": train_time_s,
    }


def compute_reward(parsed: dict) -> dict:
    checkpoints = parsed["checkpoints"]
    if not checkpoints:
        return {
            "reward": 0.0,
            "hit_target": False,
            "reason": "no val_loss checkpoints found",
            "n_val_points": 0,
        }

    step_to_target = None
    for cp in checkpoints:
        if cp["val_loss"] <= TARGET_VAL_LOSS:
            step_to_target = cp["step"]
            break

    final_val_loss = checkpoints[-1]["val_loss"]
    min_val_loss = min(c["val_loss"] for c in checkpoints)
    mean_val_loss = sum(c["val_loss"] for c in checkpoints) / len(checkpoints)

    hit_target = step_to_target is not None
    if hit_target:
        span = BASELINE_STEPS - STRETCH_STEPS
        reward = max(0.0, min(1.0, (BASELINE_STEPS - step_to_target) / span))
    else:
        reward = 0.0

    # Track-3 statsig margin at n=1 (grader operates per-trial;
    # aggregation across seeds is the outer loop's job)
    margin = TARGET_VAL_LOSS - final_val_loss
    stat_sig_at_n1 = margin * math.sqrt(1)

    return {
        "reward": round(reward, 6),
        "hit_target": hit_target,
        "step_to_3_28": step_to_target,
        "final_val_loss": final_val_loss,
        "min_val_loss": min_val_loss,
        "mean_val_loss": round(mean_val_loss, 6),
        "n_val_points": len(checkpoints),
        "final_step": checkpoints[-1]["step"],
        "total_steps_planned": parsed["total_steps"],
        "train_time_s": parsed["final_train_time_s"],
        "stat_sig_at_n1": round(stat_sig_at_n1, 6),
        "target_val_loss": TARGET_VAL_LOSS,
        "baseline_steps": BASELINE_STEPS,
    }


def grade(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    parsed = parse_log(text)
    return compute_reward(parsed)


def main(argv):
    if len(argv) == 2:
        result = grade(Path(argv[1]))
        print(json.dumps(result, indent=2))
        return 0
    if len(argv) == 4 and argv[1] == "--write":
        result = grade(Path(argv[2]))
        Path(argv[3]).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
