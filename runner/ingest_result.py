#!/usr/bin/env python3
"""Append a normalized row to runs.jsonl from a Harbor-shape trial result.

Row schema (one JSON object per line):
    variant, seed, backend, timestamp_utc, task_id, trial, status,
    reward, hit_target, step_to_3_28, final_val_loss, min_val_loss,
    mean_val_loss, n_val_points, final_step, total_steps_planned,
    train_time_s, stat_sig_at_n1, error, log_path
"""
import fcntl
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


CANONICAL_FIELDS = [
    "variant", "seed", "backend", "timestamp_utc", "task_id", "trial", "status",
    "reward", "hit_target", "step_to_3_28", "final_val_loss", "min_val_loss",
    "mean_val_loss", "n_val_points", "final_step", "total_steps_planned",
    "train_time_s", "stat_sig_at_n1", "error", "log_path", "proxy_base_url",
]


def normalize(*, variant: str, seed: int, backend: str, task_id: str, trial: int,
              status: str, reward_payload: dict, error: str | None,
              log_path: str | None, proxy_base_url: str | None = None) -> dict:
    row = {
        "variant": variant,
        "seed": seed,
        "backend": backend,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "trial": trial,
        "status": status,
        "error": error,
        "log_path": log_path,
        "proxy_base_url": proxy_base_url,
    }
    for key in ["reward", "hit_target", "step_to_3_28", "final_val_loss",
                "min_val_loss", "mean_val_loss", "n_val_points", "final_step",
                "total_steps_planned", "train_time_s", "stat_sig_at_n1"]:
        row[key] = reward_payload.get(key)
    return {k: row.get(k) for k in CANONICAL_FIELDS}


def append(ledger: Path, row: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main(argv):
    if len(argv) != 3:
        print("usage: ingest_result.py <reward.json> <runs.jsonl>", file=sys.stderr)
        return 2
    reward = json.loads(Path(argv[1]).read_text())
    row = normalize(
        variant=reward.get("_variant", "unknown"),
        seed=int(reward.get("_seed", 0)),
        backend=reward.get("_backend", "unknown"),
        task_id=reward.get("_task_id", "nanogpt/track3-speedrun"),
        trial=int(reward.get("_trial", 0)),
        status=reward.get("_status", "success" if reward.get("hit_target") else "no_target"),
        reward_payload=reward,
        error=reward.get("_error"),
        log_path=reward.get("_log_path"),
    )
    append(Path(argv[2]), row)
    print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
