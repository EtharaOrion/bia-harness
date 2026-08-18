#!/usr/bin/env python3
"""The code-driven refinement loop: build history, run harbor, parse, judge, record.

Ported from track3-pipeline/tools/refine.py (`judge_trajectory`, `run_iteration`,
`main`). Everything this module needs already exists as a tested sibling -- trial
parsing, history rendering, summarisation, judging, config building -- so this file
is only the control flow that joins them. Nothing here is interactive: a caller
passes a task and CODE drives every iteration to a ledger row.

CURRENTLY UNWIRED: THE VERIFIER-DERIVED REWARD, THE JUDGE AND THE SUMMARISER.
By default an iteration reaches no LLM and does not take its score from the task
verifier. The reward is computed by `agentloop.reward` from the val_loss curve the run
actually wrote -- `(BASELINE_STEPS - step) / (BASELINE_STEPS - TARGET_STEPS)`, clamped
to [0, 1] -- so the loop scores itself from measurement alone. `verifier/reward_full.json`
is still parsed for `reason` and `metrics` (harmless when absent); only its reward is
ignored.

This is an UNWIRING, not a removal. `judge.py` and `summariser.py` are untouched and
still tested, `judge_trajectory` and the enrichment block below are intact, and both
`run_iteration` and `refine` keep their `summarise` / `judge_enabled` parameters --
only the defaults flipped to False. Re-enable per call by passing True, or from the
CLI with the opt-IN `--summarise` / `--judge` flags. Everything in the DURABILITY note
below therefore still applies verbatim the moment either is switched back on.

Six properties are load-bearing, and each one exists because its absence cost a
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

ISOLATION. `jobs_dir` resolves under this harness's `runs/agentloop/<task-uuid>/`, never
into the read-only 65GB track3-pipeline tree the port came from.

TERMINATION. `--timeout` bounds a single harbor job and is threaded all the way to
`subprocess.run`; a wedged container is recorded as returncode 124 (GNU `timeout(1)`'s
convention) and the campaign moves on instead of hanging forever with no output.

CLEANUP. Harbor's docker containers are NOT children of the harbor process, so a
timeout or a Ctrl-C leaves them holding GPUs. Each iteration snapshots the running
`task__*` containers before and after its launch and force-removes the DIFFERENCE in a
`finally`. Only the difference, ever: this host runs concurrent campaigns, and a
blanket removal by name filter would kill someone else's live trial. Ctrl-C likewise
still writes the row for what was already measured before it re-raises -- an iteration
that consumed GPU time must leave a trace.

EXCLUSION. `resolve_run_root` is a pure function of the task, so two concurrent
`refine` calls on one task would share run_root, ledger.jsonl, the `start` each
derives from it, job_name and .cfg_iterNN.json, and interleave into each other. The
whole call therefore runs under an exclusive `flock` on `run_root/.lock`, which also
serialises ledger appends -- each of which is one `os.write` of a whole line, because
POSIX only guarantees an atomic append up to PIPE_BUF and a row carries up to 18000
chars of `parent_source`. A line that is damaged anyway is reported by line number
rather than skipped in silence: a dropped row loses an iteration AND renumbers every
one after it.

Job artifacts are the one thing this module will delete, and only on request:
`--keep-jobs N` prunes all but the newest N `jobs/agentic_iterNN/` dirs after a
successful iteration, never the one just produced, never on the interrupted path, and
never at all unless the flag is passed. Silently deleting a GPU trial would be a worse
failure than the full disk it prevents.

DIVERGENCES FROM THE SOURCE. The original resolves ROOT/RUNS/TASK/HARBOR through
module globals and takes `--config` as a path to a hand-written JSON file. This
harness is multi-task, so `run_root`, `task_dir` and `harbor_bin` are parameters and
the config is BUILT and VALIDATED programmatically by `harbor_config` instead of
being trusted from disk. `--export-traces` is also passed explicitly here; see
`run_iteration`.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):  # executed as `python runner/agentloop/loop.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import HARNESS_ROOT, resolve_harbor_bin, resolve_task  # noqa: E402
from agentloop import judge, summariser  # noqa: E402
from agentloop.harbor_config import (  # noqa: E402
    HarborUnavailable,
    build_base_cfg,
    resolve_run_root,
    validate_cfg,
)
from agentloop.history import render_history  # noqa: E402
from agentloop.trial_io import agent_findings, find_trial, read_trial, task_budget_hours  # noqa: E402

__all__ = ["IterationInterrupted", "RunRootBusy", "build_parser", "judge_trajectory",
           "load_ledger", "main", "refine", "run_iteration"]

# Exit code recorded when harbor is killed by our own timeout. 124 is what GNU
# `timeout(1)` reports, so a ledger reader needs no harness-specific convention.
TIMEOUT_RETURNCODE = 124

# 128 + SIGINT, the shell's convention for a process killed by Ctrl-C.
INTERRUPT_RETURNCODE = 130

MAX_ERROR_CHARS = 200

# Harbor derives its container names from the trial, e.g.
# `task__<id>__env-main-1` and `task__<id>__env-harbor-docker-egress-control-sidecar-1`.
CONTAINER_NAME_FILTER = "name=task__"

# Cleanup is best-effort bookkeeping around a multi-hour job; it may not become the
# thing that hangs it.
DOCKER_CMD_TIMEOUT = 30


class IterationInterrupted(KeyboardInterrupt):
    """Ctrl-C during a job, carrying the row measured up to that point.

    Subclasses KeyboardInterrupt so that a caller which does not know about it still
    sees the interrupt it asked for; `refine` catches it only to get the row onto
    disk before letting it continue upward.
    """

    def __init__(self, row: dict):
        super().__init__("iteration interrupted")
        self.row = row


class RunRootBusy(RuntimeError):
    """Another process already holds this task's run root."""


def _docker_task_containers() -> set[str] | None:
    """Ids of the running `task__*` containers, or None if docker cannot be asked.

    None and the empty set mean different things and must not be conflated: an empty
    set is evidence that nothing was running, while None is the absence of evidence,
    and only evidence may authorise a removal.
    """
    try:
        proc = subprocess.run(
            ["docker", "ps", "-q", "--filter", CONTAINER_NAME_FILTER],
            capture_output=True, text=True, timeout=DOCKER_CMD_TIMEOUT)
    except Exception:  # noqa: BLE001 - no docker, no daemon, no permission, hung cli
        return None
    if proc.returncode != 0:
        return None
    return {ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()}


def _remove_new_containers(before: set[str] | None, job_name: str) -> None:
    """Remove the containers that appeared while this iteration ran, and only those.

    Harbor's containers outlive the `harbor` process, so a timeout or a Ctrl-C
    orphans them holding GPUs. The difference between the two snapshots is the whole
    safety argument: a blanket `docker rm` of the `task__*` name filter would also
    kill a CONCURRENT campaign's live containers on this host, so a container that
    was already running before we launched is never touched -- and if the "before"
    snapshot could not be taken at all, nothing is removed.

    Every failure here is swallowed. Leaked containers cost disk and GPUs; a raised
    exception costs the iteration that just ran for hours.
    """
    if before is None:
        return
    try:
        after = _docker_task_containers()
        if after is None:
            return
        appeared = sorted(after - before)
        if not appeared:
            return
        print(f"  [cleanup] removing {len(appeared)} container(s) started by "
              f"{job_name}", file=sys.stderr, flush=True)
        subprocess.run(["docker", "rm", "-f", *appeared],
                       capture_output=True, text=True, timeout=DOCKER_CMD_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] container cleanup failed (containers may be leaked): "
              f"{type(exc).__name__}: {str(exc)[:MAX_ERROR_CHARS]}",
              file=sys.stderr, flush=True)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_ledger(ledger_path) -> list[dict]:
    """Read the JSONL ledger, tolerating the damage an interrupted run leaves behind.

    A missing file is an empty campaign, not an error. A malformed line is skipped
    rather than fatal: rows are appended after every iteration, so a kill during that
    write leaves a truncated final line, and refusing to parse the file at that point
    would strand every completed iteration before it. Order is file order, because
    that is iteration order.

    But skipping LOUDLY, naming the line. Skipping in silence was the real defect: a
    dropped row is a lost GPU-expensive iteration, and because `start = len(rows)+1`
    counts only what survived, it also silently renumbers every iteration after it --
    so the damage is invisible in the very file you would look at to find it.
    """
    path = Path(ledger_path)
    if not path.is_file():
        return []
    rows: list[dict] = []
    dropped = 0
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            dropped += 1
            print(f"  [warn] {path}:{lineno}: unreadable ledger line SKIPPED "
                  f"({type(exc).__name__}); {len(line)} chars beginning "
                  f"{line[:80]!r}", file=sys.stderr, flush=True)
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            dropped += 1
            print(f"  [warn] {path}:{lineno}: ledger line SKIPPED, expected a JSON "
                  f"object and got {type(row).__name__}", file=sys.stderr, flush=True)
    if dropped:
        print(f"  [warn] {path}: {dropped} line(s) skipped, {len(rows)} row(s) "
              f"recovered. Iteration numbering is derived from the surviving rows, "
              f"so a skipped row shifts every iteration after it -- check the file "
              f"before continuing the campaign.", file=sys.stderr, flush=True)
    return rows


@contextmanager
def run_root_lock(run_root: Path, task_desc: str):
    """Hold this run root exclusively for the duration of a campaign.

    `resolve_run_root` is a pure function of the task, so two concurrent `refine`
    calls on one task share run_root, ledger.jsonl, the `start = len(rows)+1` they
    each derive from it, job_name and .cfg_iterNN.json -- and interleave into each
    other's numbering and configs.

    There is deliberately NO --force. flock is held by an open file description, so
    the kernel drops it the moment the holder dies for any reason, including SIGKILL
    and a power cut: a lock that is still held is evidence of a LIVE run, never of a
    stale one. --force could therefore only ever override a genuinely running
    campaign, reinstating exactly the corruption this exists to prevent. If a lock
    ever does look stuck, `fuser run_root/.lock` names the process that holds it.
    """
    lock_path = Path(run_root) / ".lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RunRootBusy(
                f"another refine run is already active for task {task_desc}: "
                f"{lock_path} is locked. Two runs of one task share its run root "
                f"({run_root}), its ledger and its iteration numbering, so this one "
                f"is refused. Wait for the other run to finish, or check it with "
                f"`fuser {lock_path}`."
            ) from exc
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()} {utc()}\n".encode())
            yield lock_path
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _append_row(ledger: Path, row: dict) -> None:
    """Append one row so that it can never be observed half-written.

    A row carries up to 18000 chars of `parent_source`, and POSIX guarantees an
    O_APPEND write is atomic only up to PIPE_BUF (4096 bytes) -- so the buffered
    `f.write()` this replaces could split a row across two writes and interleave with
    another appender. Two things prevent that here: the whole line leaves in a single
    `os.write` (the loop only ever repeats on a short write, which O_APPEND makes
    vanishingly rare), and every writer holds the run root lock (`run_root_lock`),
    which serialises them regardless of size. The fsync is for the other half of the
    problem: a row still sitting in the page cache when the machine dies did not
    survive the iteration it was meant to record.
    """
    payload = (json.dumps(row, sort_keys=True, default=str) + "\n").encode()
    fd = os.open(ledger, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def _prune_jobs(jobs_dir, keep: int | None, *, protect: str) -> None:
    """Keep the newest `keep` job dirs and delete the rest. `None` deletes nothing.

    Harbor trial artifacts are gigabytes per GPU trial and accumulate forever, and a
    full disk crashes the next iteration's writes. But this deletes measured results,
    so it happens only when asked for, only after the row it belongs to is durably on
    disk, and never to the dir this iteration just produced -- which is protected by
    name rather than by trusting its mtime to sort first.

    Every failure is swallowed: pruning is housekeeping, and housekeeping may not
    cost the iteration that just ran for hours.
    """
    if keep is None:
        return
    try:
        jobs_dir = Path(jobs_dir)
        if not jobs_dir.is_dir():
            return
        newest_first = sorted((d for d in jobs_dir.iterdir() if d.is_dir()),
                              key=lambda d: (d.stat().st_mtime, d.name), reverse=True)
        doomed = [d for d in newest_first[max(keep, 0):] if d.name != protect]
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not list {jobs_dir} to prune it: "
              f"{type(exc).__name__}: {str(exc)[:MAX_ERROR_CHARS]}",
              file=sys.stderr, flush=True)
        return
    for d in doomed:
        try:
            shutil.rmtree(d)
            print(f"  [prune] removed old job dir {d}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] could not prune {d}: {type(exc).__name__}: "
                  f"{str(exc)[:MAX_ERROR_CHARS]}", file=sys.stderr, flush=True)


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
                  summarise: bool = False, judge_enabled: bool = False,
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
    # Snapshot before launching so the finally below can remove exactly what this
    # iteration added. Taken after validation, so a rejected config launches nothing
    # and asks docker nothing.
    containers_before = _docker_task_containers()
    interrupted = False
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
    except KeyboardInterrupt:
        # Ctrl-C stops the campaign, but not before the hours already spent are
        # parsed and recorded. The interrupt is re-raised at the end of this
        # function, wrapped around the finished row.
        print(f"\n  [warn] interrupted during {job_name}; recording what was "
              f"measured", file=sys.stderr, flush=True)
        interrupted = True
        returncode = INTERRUPT_RETURNCODE
    finally:
        # Harbor's containers are not children of the harbor process, so neither a
        # timeout nor a Ctrl-C reaps them; without this they sit on the GPUs.
        _remove_new_containers(containers_before, job_name)

    job_dir = Path(cfg["jobs_dir"]) / job_name
    trial = find_trial(job_dir, t0) if job_dir.is_dir() else None
    if trial is None:
        # Absence is data. A launch that produced nothing is still an iteration and
        # still belongs in the ledger, or the next attempt sees a gap it cannot explain.
        sparse = {"iteration": i, "reward": 0.0, "reason": "no_trial_produced",
                  "outcome": "harness_incomplete", "n_seeds": 0, "findings": "",
                  "timestamp_utc": utc(), "job_name": job_name,
                  "harbor_returncode": returncode}
        if interrupted:
            sparse["interrupted"] = True
            raise IterationInterrupted(sparse)
        return sparse

    row = read_trial(trial)
    row.update({"iteration": i, "timestamp_utc": utc(), "job_name": job_name,
                "findings": agent_findings(trial), "harbor_returncode": returncode,
                "history_injected": bool(history)})
    if interrupted:
        row["interrupted"] = True

    # DURABILITY CHECKPOINT. Everything above is measured fact; everything below is
    # advisory LLM enrichment that reaches over the network. The facts are put on
    # disk first so they survive not only an exception (which the handlers below
    # catch) but a SIGKILL, an OOM or a pulled plug, which no handler can catch.
    (history_dir / f"iter{i:02d}_facts.json").write_text(
        json.dumps(row, indent=2, default=str))

    if interrupted:
        # Enrichment is advisory and reaches over the network; an operator who
        # pressed Ctrl-C is not waiting for it. The facts are complete and durable.
        raise IterationInterrupted(row)

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
           summarise: bool = False, judge_enabled: bool = False,
           harbor_bin: str | None = None,
           agent_name: str = "claude-code",
           base_cfg_overrides: dict | None = None,
           timeout: float | None = None,
           keep_jobs: int | None = None) -> list[dict]:
    """Drive `iterations` refinement rounds over `task`, appending each to the ledger.

    Returns every row the ledger holds afterwards, prior rows included, because the
    campaign -- not this call -- is the unit of meaning.
    """
    task_dir = resolve_task(str(task))
    run_root = resolve_run_root(task_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    ledger = run_root / "ledger.jsonl"
    harbor_bin = harbor_bin or resolve_harbor_bin()

    # Everything that reads or writes the run root happens under this lock, including
    # the `start` derivation: taking it any later would leave the read of the ledger
    # racing the other run's append.
    with run_root_lock(run_root, task_desc=str(task_dir)):
        base_cfg = build_base_cfg(task_dir, run_root, agent_name=agent_name)
        if base_cfg_overrides:
            base_cfg.update(base_cfg_overrides)

        rows = load_ledger(ledger)
        # The ledger IS the loop state. Deriving `start` from it -- rather than from a
        # counter held in memory or a separate state file -- is what makes an
        # interrupted campaign resumable by simply running the same command again.
        start = start_at if start_at is not None else len(rows) + 1
        budget = task_budget_hours(task_dir)

        for n in range(iterations):
            i = start + n
            history = render_history(rows, total_iterations=start + iterations - 1,
                                     budget_hours=budget)
            try:
                row = run_iteration(i, base_cfg, history, run_root=run_root,
                                    task_dir=task_dir, harbor_bin=harbor_bin,
                                    summarise=summarise, judge_enabled=judge_enabled,
                                    timeout=timeout)
            except IterationInterrupted as exc:
                # An iteration that consumed GPU time must leave a trace. The append
                # used to sit only on the success path, so Ctrl-C during a multi-hour
                # job discarded everything it had already measured.
                _append_row(ledger, exc.row)
                raise
            # Appended IMMEDIATELY, before any further work: an iteration that is not
            # on disk did not happen as far as a resumed campaign is concerned.
            _append_row(ledger, row)
            rows.append(row)
            print(f"[{utc()}] iteration {i}: reward={row.get('reward', 0.0):.4f} "
                  f"outcome={row.get('outcome')} seeds={row.get('n_seeds')} "
                  f"tokens_in={row.get('n_input_tokens')}", flush=True)
            # After the append, never before: pruning must not be able to cost the
            # row that pays for it.
            _prune_jobs(base_cfg.get("jobs_dir"), keep_jobs,
                        protect=str(row.get("job_name") or ""))

    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Code-driven refinement loop: run a task N times, feeding each "
                    "iteration's summarised history into the next.")
    ap.add_argument("--task", required=True,
                    help="task name, uuid or path under tasks/")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--start-at", type=int, default=None,
                    help="override the iteration number derived from the ledger")
    ap.add_argument("--summarise", action="store_true",
                    help="EXPERIMENTAL, currently unwired: opt in to the LLM "
                         "summariser. Off by default; the loop records measured "
                         "facts only and reaches no LLM.")
    ap.add_argument("--judge", action="store_true",
                    help="EXPERIMENTAL, currently unwired: opt in to the LLM "
                         "trajectory grader. Off by default; its verdict is advisory "
                         "and never moves the reward.")
    ap.add_argument("--agent", choices=["claude-code", "openhands-sdk"],
                    default="claude-code",
                    help="harbor agent to drive (default: claude-code). "
                         "openhands-sdk reaches the SAME Claude OAuth bridge, via "
                         "LiteLLM with an anthropic/-prefixed model.")
    ap.add_argument("--harbor-bin", default=None,
                    help="harbor executable (default: resolved from $HARBOR_BIN, "
                         "the sibling .venv-harbor, then PATH)")
    ap.add_argument("--timeout", type=float, default=None,
                    help="seconds to let one harbor job run before killing it and "
                         "recording harbor_returncode=124 (default: None, i.e. NO "
                         "limit -- a wedged container would otherwise hang the "
                         "campaign forever)")
    ap.add_argument("--keep-jobs", type=int, default=None, metavar="N",
                    help="after each successful iteration, DELETE all but the newest "
                         "N job dirs under run_root/jobs (default: unset, i.e. keep "
                         "ALL of them and delete nothing). The dir just produced is "
                         "never deleted.")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    rows = refine(args.task, iterations=args.iterations, start_at=args.start_at,
                  summarise=args.summarise,
                  judge_enabled=args.judge,
                  harbor_bin=args.harbor_bin,
                  agent_name=args.agent,
                  timeout=args.timeout,
                  keep_jobs=args.keep_jobs)

    print("\n=== summary ===")
    print(f"  iterations in ledger : {len(rows)}")
    # Only rows that actually reached the target loss are eligible to be "best": a
    # reward of 0.0 means no seed crossing was measured at all, not a poor result.
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
