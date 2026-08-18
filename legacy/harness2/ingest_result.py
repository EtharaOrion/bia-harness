#!/usr/bin/env python3
"""Append a normalized row to runs.jsonl from a Harbor-shape trial result.

Row schema (one JSON object per line, 33 canonical fields):
    variant, attempt, seed, backend, timestamp_utc, task_id, trial, status,
    reward, hit_target, step_to_3_28, final_val_loss, min_val_loss,
    mean_val_loss, n_val_points, final_step, total_steps_planned,
    train_time_s, stat_sig_at_n1, input_tokens, output_tokens, total_tokens,
    error, log_path, proxy_base_url, work_dir, harbor_trial_dir,
    verifier_dir, consolidated_reward, verification_complete, process_score,
    model_name, retry_count

One row = one seed of one LLM-authored variant. `attempt` = 1-indexed
iteration number (a.k.a. RFP "attempt" or PI "variant"), `seed` = RNG
index within that attempt. Rows written from single-shot `run()` have
attempt=None; rows from the LLM refinement loop `orchestrator.run_loop`
have attempt >= 1. `work_dir` is the absolute path of the per-seed
work directory (SPEC_AUDIT §6.5, R17). `harbor_trial_dir` is the
per-seed copytree of Harbor's trial_dir (SPEC_AUDIT R13/D17); None
for non-harbor backends. `verifier_dir`, `consolidated_reward`,
`verification_complete`, `process_score` are populated by BIA verifier
(SPEC_AUDIT R1, D1, D2); None when BIA did not run (no verifier assets
or single-shot run with attempt_dir=None). `model_name` is the LLM
model actually used to author the variant this row belongs to (populated
by orchestrator.run_loop from llm_config['model']; None for single-shot
run() with no LLM). `retry_count` is 0-indexed infra-retry attempt
number for this seed (SPEC_AUDIT R12, G2); default 0 until --retry K
is wired.
"""
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path


CANONICAL_FIELDS = [
    "variant", "attempt", "seed", "backend", "timestamp_utc", "task_id", "trial",
    "status", "reward", "hit_target", "step_to_3_28", "final_val_loss",
    "min_val_loss", "mean_val_loss", "n_val_points", "final_step",
    "total_steps_planned", "train_time_s", "stat_sig_at_n1",
    "input_tokens", "output_tokens", "total_tokens",
    "error", "log_path", "proxy_base_url", "work_dir", "harbor_trial_dir",
    "verifier_dir", "consolidated_reward", "verification_complete", "process_score",
    "model_name", "retry_count",
]


def normalize(*, variant: str, seed: int, backend: str, task_id: str, trial: int,
              status: str, reward_payload: dict, error: str | None,
              log_path: str | None, proxy_base_url: str | None = None,
              attempt: int | None = None,
              input_tokens: int | None = None,
              output_tokens: int | None = None,
              total_tokens: int | None = None,
              work_dir: str | None = None,
              harbor_trial_dir: str | None = None,
              verifier_dir: str | None = None,
              consolidated_reward: float | None = None,
              verification_complete: bool | None = None,
              process_score: float | None = None,
              model_name: str | None = None,
              retry_count: int = 0) -> dict:
    row = {
        "variant": variant,
        "attempt": attempt,
        "seed": seed,
        "backend": backend,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "trial": trial,
        "status": status,
        "error": error,
        "log_path": log_path,
        "proxy_base_url": proxy_base_url,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "work_dir": work_dir,
        "harbor_trial_dir": harbor_trial_dir,
        "verifier_dir": verifier_dir,
        "consolidated_reward": consolidated_reward,
        "verification_complete": verification_complete,
        "process_score": process_score,
        "model_name": model_name,
        "retry_count": retry_count,
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
