#!/usr/bin/env python3
"""The code-driven refinement loop: build history, run harbor, parse, judge, record.

Ported from track3-pipeline/tools/refine.py (`judge_trajectory`, `run_iteration`,
`main`). Everything this module needs already exists as a tested sibling -- trial
parsing, history rendering, summarisation, judging, config building -- so this file
is only the control flow that joins them. Nothing here is interactive: a caller
passes a task and CODE drives every iteration to a ledger row.

Three properties are load-bearing, and each one exists because its absence cost a
real run.

DURABILITY. The judge and the summariser are ADVISORY. Neither can move a reward,
so neither may cost an iteration. `judge.grade_attempt` raises `SystemExit` at two
sites by design, and `SystemExit` does not inherit from `Exception` -- an
`except Exception` around it catches nothing. In the reference pipeline that exact
gap destroyed a completed 71-minute, $9.17 trial: the judge 403'd, `SystemExit`
unwound the loop, and the process died BEFORE the ledger row was written. Every
measured fact was lost to preserve a verdict that would have been discarded anyway.
So enrichment is wrapped in `except BaseException`, and the measured facts are
checkpointed to disk BEFORE enrichment is attempted -- belt and braces, because the
`except` only survives an exception while the checkpoint also survives a SIGKILL.

RESUMABILITY. The ledger IS the loop state. `start` is derived from its length and
each row is appended the moment it is produced, so an interrupted campaign restarts
where it stopped with no separate bookkeeping file to fall out of sync.

ISOLATION. `jobs_dir` resolves under this harness's `runs/track3/<task-uuid>/`, never
into the read-only 65GB track3-pipeline tree the port came from.

DIVERGENCES FROM THE SOURCE. The original resolves ROOT/RUNS/TASK/HARBOR through
module globals and takes `--config` as a path to a hand-written JSON file. This
harness is multi-task, so `run_root`, `task_dir` and `harbor_bin` are parameters and
the config is BUILT and VALIDATED programmatically by `harbor_config` instead of
being trusted from disk. `--export-traces` is also passed explicitly here; see
`run_iteration`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):  # executed as `python runner/track3/loop.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import HARNESS_ROOT, resolve_harbor_bin, resolve_task  # noqa: E402
from track3 import judge, summariser  # noqa: E402
from track3.harbor_config import (  # noqa: E402
    HarborUnavailable,
    build_base_cfg,
    resolve_run_root,
    validate_cfg,
)
from track3.history import render_history  # noqa: E402
from track3.trial_io import agent_findings, find_trial, read_trial, task_budget_hours  # noqa: E402

__all__ = ["judge_trajectory", "load_ledger", "main", "refine", "run_iteration"]

# Exit code recorded when harbor is killed by our own timeout. 124 is what GNU
# `timeout(1)` reports, so a ledger reader needs no harness-specific convention.
TIMEOUT_RETURNCODE = 124

MAX_ERROR_CHARS = 200


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_ledger(ledger_path) -> list[dict]:
    """Read the JSONL ledger, tolerating the damage an interrupted run leaves behind.

    A missing file is an empty campaign, not an error. A malformed line is skipped
    rather than fatal: rows are appended after every iteration, so a kill during that
    write leaves a truncated final line, and refusing to parse the file at that point
    would strand every completed iteration before it. Order is file order, because
    that is iteration order.
    """
    path = Path(ledger_path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def judge_trajectory(trial: Path, model: str = "claude-opus-5") -> dict:
    """The LLM trajectory grader's verdict, recorded beside the reward.

    Veto-only and advisory: the verdict is never blended into `reward`, so a judge
    error cannot move a score, and a judge outage must not fail an otherwise complete
    iteration.
    """
    try:
        verdict = judge.grade_attempt(trial, model=model)
        failed = [k for k, r in (verdict.get("verdicts") or {}).items()
                  if not (r or {}).get("pass")]
        return {"overall_pass": verdict.get("overall_pass"),
                "summary": verdict.get("summary"),
                "failed_rubrics": failed}
    except BaseException as e:  # noqa: BLE001
        # BaseException, NOT Exception, and this is deliberate -- do not "fix" it.
        # `judge.grade_attempt` raises SystemExit at two sites (bridge unreachable,
        # empty input) and SystemExit inherits from BaseException, so `except
        # Exception` here would catch nothing and a judge outage would unwind the
        # whole loop -- discarding a completed, already-graded trial to save an
        # advisory verdict. That is precisely how a 71-minute $9.17 run was lost.
        return {"error": f"{type(e).__name__}: {str(e)[:MAX_ERROR_CHARS]}"}


def run_iteration(i: int, base_cfg: dict, history: str, *,
                  run_root: Path, task_dir: Path, harbor_bin: str,
                  summarise: bool = True, judge_enabled: bool = True,
                  timeout: float | None = None) -> dict:
    """Run one harbor job and return the ledger row describing it.

    Returns a row for every outcome, including a run that produced no trial at all.
    Raising instead would lose the fact that the iteration happened.
    """
    run_root = Path(run_root)
    history_dir = run_root / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    # Deep copy via JSON, not copy.deepcopy: it round-trips through the same
    # serialisation harbor will read, so a value that cannot survive that trip fails
    # here rather than silently differing from what the job actually ran with. Also
    # keeps `base_cfg` pristine, so iteration 3's job_name is `agentic_iter03` and
    # not `agentic_iter01_iter02_iter03`.
    cfg = json.loads(json.dumps(base_cfg))
    job_name = f"{base_cfg['job_name']}_iter{i:02d}"
    cfg["job_name"] = job_name

    if history:
        hp = history_dir / f"iter{i:02d}_history.md"
        hp.write_text(history)
        cfg["extra_instruction_paths"] = [str(hp)]

    # Validate BEFORE launching. This is the cheapest possible check and the only
    # one that happens before GPU time is spent; harbor itself would ignore an
    # unknown key and run happily with the setting you thought you set.
    try:
        validate_cfg(cfg)
    except HarborUnavailable as exc:
        # Not being able to validate is not evidence of an invalid config. Harbor's
        # venv may be absent on a host that is only rehearsing the loop, and hard
        # failing here would make the module unusable there. A genuinely broken
        # config still fails at launch, one step later.
        print(f"  [warn] skipping cfg validation: {exc}", file=sys.stderr, flush=True)

    cfg_path = run_root / f".cfg_iter{i:02d}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    print(f"[{utc()}] iteration {i}: launching harbor job {job_name}", flush=True)
    t0 = time.time()
    cmd = [str(harbor_bin), "run", "--config", str(cfg_path), "--export-traces"]
    # --export-traces MUST be a CLI flag. JobConfig is pydantic extra="ignore", so an
    # `export_traces` key in the JSON would be silently DROPPED: the config would look
    # correct while harbor wrote no agent/trajectory.json, leaving the summariser and
    # the judge with nothing to read and every seed flagged verification_incomplete.
    try:
        proc = subprocess.run(cmd, cwd=str(HARNESS_ROOT), capture_output=True,
                              text=True, timeout=timeout)
        returncode = proc.returncode
        if returncode != 0:
            print((proc.stdout or "")[-2000:], file=sys.stderr)
            print((proc.stderr or "")[-2000:], file=sys.stderr)
    except subprocess.TimeoutExpired:
        # A timed-out job may still have written a partial trial worth parsing, so
        # this is recorded and the iteration continues rather than aborting.
        print(f"  [warn] harbor exceeded timeout={timeout}s", file=sys.stderr,
              flush=True)
        returncode = TIMEOUT_RETURNCODE

    job_dir = Path(cfg["jobs_dir"]) / job_name
    trial = find_trial(job_dir, t0) if job_dir.is_dir() else None
    if trial is None:
        # Absence is data. A launch that produced nothing is still an iteration and
        # still belongs in the ledger, or the next attempt sees a gap it cannot explain.
        return {"iteration": i, "reward": 0.0, "reason": "no_trial_produced",
                "outcome": "harness_incomplete", "n_seeds": 0, "findings": "",
                "timestamp_utc": utc(), "job_name": job_name,
                "harbor_returncode": returncode}

    row = read_trial(trial)
    row.update({"iteration": i, "timestamp_utc": utc(), "job_name": job_name,
                "findings": agent_findings(trial), "harbor_returncode": returncode,
                "history_injected": bool(history)})

    # DURABILITY CHECKPOINT. Everything above is measured fact; everything below is
    # advisory LLM enrichment that reaches over the network. The facts are put on
    # disk first so they survive not only an exception (which the handlers below
    # catch) but a SIGKILL, an OOM or a pulled plug, which no handler can catch.
    (history_dir / f"iter{i:02d}_facts.json").write_text(
        json.dumps(row, indent=2, default=str))

    if judge_enabled:
        row["rubric_verdicts"] = judge_trajectory(trial)

    if summarise:
        try:
            row["summary"] = summariser.summarize_iteration(trial, row)
        except BaseException as e:  # noqa: BLE001
            # BaseException for the same reason as in `judge_trajectory`: the
            # summariser sits behind the same bridge and can exit rather than raise.
            # A missing summary costs the next prompt some prose; a propagated one
            # costs the entire iteration.
            row["summary"] = {
                "_degraded": f"summariser_failed: {type(e).__name__}: "
                             f"{str(e)[:MAX_ERROR_CHARS]}"
            }

    return row


def refine(task, iterations: int = 3, *, start_at: int | None = None,
           summarise: bool = True, judge_enabled: bool = True,
           harbor_bin: str | None = None,
           base_cfg_overrides: dict | None = None) -> list[dict]:
    """Drive `iterations` refinement rounds over `task`, appending each to the ledger.

    Returns every row the ledger holds afterwards, prior rows included, because the
    campaign -- not this call -- is the unit of meaning.
    """
    task_dir = resolve_task(str(task))
    run_root = resolve_run_root(task_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    ledger = run_root / "ledger.jsonl"
    harbor_bin = harbor_bin or resolve_harbor_bin()

    base_cfg = build_base_cfg(task_dir, run_root)
    if base_cfg_overrides:
        base_cfg.update(base_cfg_overrides)

    rows = load_ledger(ledger)
    # The ledger IS the loop state. Deriving `start` from it -- rather than from a
    # counter held in memory or a separate state file -- is what makes an interrupted
    # campaign resumable by simply running the same command again.
    start = start_at if start_at is not None else len(rows) + 1
    budget = task_budget_hours(task_dir)

    for n in range(iterations):
        i = start + n
        history = render_history(rows, total_iterations=start + iterations - 1,
                                 budget_hours=budget)
        row = run_iteration(i, base_cfg, history, run_root=run_root,
                            task_dir=task_dir, harbor_bin=harbor_bin,
                            summarise=summarise, judge_enabled=judge_enabled)
        # Appended IMMEDIATELY, before any further work: an iteration that is not on
        # disk did not happen as far as a resumed campaign is concerned.
        with ledger.open("a") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        rows.append(row)
        print(f"[{utc()}] iteration {i}: reward={row.get('reward', 0.0):.4f} "
              f"outcome={row.get('outcome')} seeds={row.get('n_seeds')} "
              f"tokens_in={row.get('n_input_tokens')}", flush=True)

    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Code-driven refinement loop: run a task N times, feeding each "
                    "iteration's summarised history into the next.")
    ap.add_argument("--task", required=True,
                    help="task name, uuid or path under tasks/")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--start-at", type=int, default=None,
                    help="override the iteration number derived from the ledger")
    ap.add_argument("--no-summarise", action="store_true",
                    help="skip the LLM summariser; record verifier facts only")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the LLM trajectory grader; record verifier facts only")
    ap.add_argument("--harbor-bin", default=None,
                    help="harbor executable (default: resolved from $HARBOR_BIN, "
                         "the sibling .venv-harbor, then PATH)")
    args = ap.parse_args(argv)

    rows = refine(args.task, iterations=args.iterations, start_at=args.start_at,
                  summarise=not args.no_summarise,
                  judge_enabled=not args.no_judge,
                  harbor_bin=args.harbor_bin)

    print("\n=== summary ===")
    print(f"  iterations in ledger : {len(rows)}")
    # Only rows the verifier actually scored are eligible to be "best": a reward of
    # 0.0 from a run that never reached the verifier measures nothing.
    graded = [r for r in rows if (r.get("reward") or 0.0) > 0]
    if graded:
        best = max(graded, key=lambda r: r["reward"])
        print(f"  best   : iteration {best.get('iteration')} "
              f"reward {best['reward']:.4f}")
    else:
        print("  best   : none scored above 0.0")
    if rows:
        last = rows[-1]
        print(f"  last   : iteration {last.get('iteration')} "
              f"reward {(last.get('reward') or 0.0):.4f}")
    else:
        print("  last   : ledger is empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
