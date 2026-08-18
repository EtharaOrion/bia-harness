# Pipeline flow

End-to-end lifecycle of one candidate variant across any registered task.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Policy read (session start)                                         │
│     Orchestrator LLM loads policy/{AGENTS.md, goal.md, plan.md,         │
│     scratchpad/THREAD.md, picklist.md, audits.md}                       │
│                                                                         │
│  2. Task selection                                                      │
│     Orchestrator picks: (a) which task (name under tasks/ or a path),   │
│     and (b) optionally which candidate variant.                         │
│                                                                         │
│  3. Variant authoring (optional)                                        │
│     Orchestrator writes/edits scratchpad/variants/<slug>.py.            │
│     Not every task takes a variant — depends on task's mount.toml.      │
│                                                                         │
│  4. Dispatch (runner/harness.py)                                        │
│     python runner/harness.py                                            │
│       --task <name|path> [--variant <path>]                             │
│       --attempts A --seeds S --backend {harbor,local,dry}               │
│                                                                         │
│     4a. resolve_task(name) -> tasks/<name>/ or verbatim path            │
│     4b. read_task_id(task_dir) -> task.toml [task].name                 │
│     4c. mount_task(task_dir, work/task, variant=variant)                │
│         - shutil.copytree the task dir                                  │
│         - read mount.toml (optional)                                    │
│         - for each [[shared]]: copy shared/<src> -> task/<dst>          │
│         - if variant provided: copy -> task/<[variant].dst>             │
│         - variant provided but task has no [variant] block: ValueError  │
│         - [variant].required=true and no variant provided: ValueError   │
│                                                                         │
│     4d. dispatch_<backend>(mounted_task, seed, work, run_name)          │
│         harbor: `harbor run -p <mounted> -a bash -n 1 --env docker`     │
│         local:  `torchrun --standalone --nproc_per_node=N               │
│                  <mounted>/environment/train_gpt_simple.py`             │
│         dry:    fabricate synthetic Track-3-style log (wiring test)     │
│                                                                         │
│  5. Training executes (real) or synthetic log written (dry)             │
│     Log at work/runs/<run_name>_seed<N>.log                             │
│                                                                         │
│  6. Grade (each task's tests/grader.py, task-specific!)                 │
│     nanogpt-speedrun: regex on step/val_loss, reward from step_to_3.28  │
│     nanogpt-smoke:    regex on `Python X.Y` + `torch:V`, reward=1.0     │
│                       iff both present                                  │
│     Runner shells: `python <mounted>/tests/grader.py <log>` -> JSON     │
│                                                                         │
│  7. Ingest (runner/legacy_planner/ingest_result.py)                                    │
│     Normalize reward.json + metadata into 20-field canonical row.       │
│     task_id populated from task.toml (not hardcoded).                   │
│     Append one JSON line to runs.jsonl.                                 │
│                                                                         │
│  8. Loop back                                                           │
│     Orchestrator reads latest runs.jsonl (filter by task_id — one       │
│     file, many tasks), updates plan.md, applies AGENTS.md gates,        │
│     picks next candidate.                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

`--attempts` counts outer LLM-authored candidates; `--iterations` is a
deprecated alias. `--seeds` defaults to 2 independent replicas per attempt.
`--llm-retries` defaults to 3 and applies only to planner API 429/timeout
failures. It never retries training, changes the seed count, or creates an
additional attempt. The former `--retries` flag has been removed because it
mixed unrelated retry domains.

## Code-driven refinement flow (`runner/track3`)

The flow above injects an LLM-authored variant from *outside* the container. That
depends on a `[variant]` block in `mount.toml`, so it cannot be used for a task whose
`/workspace` ships inside the docker image. `runner/track3` is the other direction:
the agent authors `/workspace/submission/optimizer.py` and runs training *inside* the
container, and our code drives everything around it.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Resolve task + run root                                             │
│     resolve_task(name|uuid|path); run root = runs/track3/<slug>/        │
│     start = len(ledger rows) + 1   (or --start-at)                      │
│                                                                         │
│  2. Render history (track3/history.py)                                  │
│     prior ledger rows -> agent-facing markdown (facts table, per-       │
│     iteration prose, best-so-far source, synthetic banner if any).      │
│     Empty on iteration 1: nothing to inject, no file written.           │
│     -> history/iterNN_history.md                                        │
│                                                                         │
│  3. Build + validate config (track3/harbor_config.py)                   │
│     build_base_cfg -> job_name, jobs_dir (under the run root), agents[] │
│     with model_name/env, tasks[].path, retry policy.                    │
│     --agent picks who authors in-container. Both hit the SAME bridge;   │
│       claude-code   -> ANTHROPIC_BASE_URL/_API_KEY, model bare          │
│       openhands-sdk -> LLM_BASE_URL/LLM_API_KEY, model anthropic/-      │
│                        prefixed (LiteLLM needs it to route).            │
│     history path attached as extra_instruction_paths.                   │
│     validate_cfg checks it against harbor's REAL JobConfig and rejects  │
│     unknown keys at every depth (harbor itself would ignore them).      │
│     -> .cfg_iterNN.json                                                 │
│                                                                         │
│  4. Launch                                                              │
│     harbor run --config .cfg_iterNN.json --export-traces                │
│     --export-traces is a CLI FLAG, not a config key (see below).        │
│                                                                         │
│  5. Locate + parse the trial (track3/trial_io.py)                       │
│     find_trial(jobs_dir/job_name, since=launch time), then read_trial:  │
│     tokens + cost from result.json, submitted optimizer.py and val_loss │
│     curve from artifacts/, reason/metrics from verifier/reward_full.json│
│     (NOT reward.json). Every reader falls back on error: a partial      │
│     trial must yield a reward-0 row, not a traceback.                   │
│                                                                         │
│  5b. Score from the curve (track3/reward.py) — VERIFIER UNWIRED         │
│     reward = clamp01((3500 - step) / 600) where `step` is the first at  │
│     which EVERY seed reaches val_loss <= 3.28 (max across seeds; None   │
│     if any seed never does). The verifier's own reward is ignored.      │
│                                                                         │
│  6. Classify (track3/classify.py)                                       │
│     graded_pass | graded_miss | gate_fail | agent_abandoned_run |       │
│     harness_incomplete | unknown                                        │
│                                                                         │
│  7. Checkpoint facts -> history/iterNN_facts.json                       │
│     BEFORE any LLM enrichment, so a SIGKILL cannot lose measured facts. │
│                                                                         │
│  8. Enrich (advisory, optional) — CURRENTLY UNWIRED, OFF BY DEFAULT     │
│     judge.grade_attempt (veto-only verdict) and                         │
│     summariser.summarize_iteration (prose for the next prompt).         │
│     Both wrapped in `except BaseException` — they raise SystemExit, and │
│     neither may cost a completed iteration.                             │
│     Opt IN with --judge / --summarise; a bare run reaches no LLM.       │
│                                                                         │
│  9. Append one row to ledger.jsonl, then loop back to step 2            │
└─────────────────────────────────────────────────────────────────────────┘
```

```bash
python runner/track3/loop.py --task <name|uuid|path> --iterations N \
  [--start-at N] [--summarise] [--judge] [--harbor-bin PATH]
```

Per-task layout under `runs/track3/<slug>/`:

| Path | Contents |
|---|---|
| `ledger.jsonl` | one JSON row per iteration, appended immediately |
| `history/iterNN_history.md` | the markdown injected into iteration NN |
| `history/iterNN_facts.json` | measured facts, written before LLM enrichment |
| `.cfg_iterNN.json` | the exact config handed to `harbor run` |
| `jobs/<job_name>/<trial>/` | harbor's own trial output |

`<slug>` is the task uuid, or a slug of `[task].name` when the bundle declares none.

**The ledger is the loop state.** `start` is derived from its length, and each row is
written the moment it exists, so an interrupted campaign resumes rather than restarts —
re-running the same command continues at the next iteration. A truncated final line
(the normal shape of a kill mid-write) is skipped, not fatal.

**`--export-traces` is a CLI flag by necessity.** `JobConfig` is pydantic
`extra="ignore"`, so harbor silently DROPS keys it does not define. An `export_traces`
key in the JSON would produce a config that looks correct while no
`agent/trajectory.json` is written, leaving the summariser and judge with nothing to
read and every seed flagged `verification_incomplete`.

**Synthetic rows never pass as real.** `--backend dry` fabricates its curve from a seed
hash; those rows carry `is_synthetic`, and `history.render_history` prints
`marking.SYNTHETIC_BANNER` above the facts table when any shown row is synthetic. They
are not validated results and must not be reported as such.

`tests/test_track3_integration.py` exercises this against real harbor and real docker,
but only under `TRACK3_INTEGRATION=1` (plus a harbor binary, a live docker daemon and
the task image); otherwise it skips, because one iteration starts a GPU container.

## BIA verifier flow

`bia_verifier` is an explicit post-run verification pipeline rather than a
hidden part of metric extraction:

1. Load the dataset truth path and concern registry; reject a non-MECE spec.
2. Discover submission, logs, telemetry, trajectory, and observed reward.
3. Execute trusted deterministic predicates once.
4. Validate that rubric IDs exactly match rubric-owned concerns.
5. Judge each rubric item once with `gpt5.6-sol` through the authenticated local
   Codex bridge, or load a complete precomputed judgement file.
6. Produce one report/reward/outcomes/rubric artifact set from those same
   results.
7. Generate deterministic pytest assertions, execute them against
   `outcomes.json`, and write `pytest.xml` plus `pytest_report.json`.

Generated pytest failure makes the verifier CLI exit nonzero. Judge outages or
unparseable replies are recorded as `not_measured`, not fabricated failures.
The verifier is currently an explicit post-run command; the task runner does
not automatically invoke it per seed yet. See `BIA_VERIFIER.md` for the full
contract and the ordering required when that integration is added.

## Task registry

Tasks live under `tasks/<name>/` and each is a self-contained Harbor task.
Runner discovers them via `resolve_task`:

- `--task nanogpt-speedrun` → `<harness_root>/tasks/nanogpt-speedrun/`
- `--task ./my_experimental_task` → verbatim relative path
- `--task /abs/path/to/task` → verbatim absolute path

Missing → `FileNotFoundError` with both attempted locations printed.

## mount.toml schema

Every task MAY have a `mount.toml`. If absent, the task is copied verbatim,
no injection.

```toml
version = "1.0"

[[shared]]
src = "shared/<file>"        # path relative to harness_root
dst = "environment/<file>"   # path inside the mounted task dir

[[shared]]  # repeat for each shared asset
...

[variant]                    # optional
dst = "environment/<file>"   # where --variant lands
required = false             # if true, runner errors when --variant absent
```

Constraints (enforced by test_mount_toml.py):
- `[[shared]].src` must exist at `<harness_root>/<src>` at run time.
- `[[shared]].dst` and `[variant].dst` should stay inside task subdirs
  (`environment/`, `tests/`, or `solution/`).

## Backend behaviors

### `--backend dry` (CPU wiring test)

Fabricates a synthetic Track-3-style training log (12 val_loss checkpoints,
linear 4.0 → 1.25 over 3000 steps). Purpose: exercise mount → dispatch →
grade → ingest without GPUs or Docker.

**Known limitation** (by design): the synthetic log format is Track-3-shaped.
A task whose grader expects a different log format (e.g. nanogpt-smoke's
grader looks for `Python X.Y` and `torch:V` lines) will correctly return
`reward=0.0` under `--backend dry`. That's expected: dry backend proves
plumbing, not task-specific grading. Real grading uses `--backend local`
or `--backend harbor`.

### `--backend local` (single GPU host, no Harbor)

Requires: `nvidia-smi` in PATH, torch installed, CUDA runtime, cached data
under the mounted task's `environment/` (or a task that populates its own data).

Runs `torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l)
<mounted>/environment/train_gpt_simple.py` with `cwd=<mounted>/environment/`.

If mount didn't populate `environment/train_gpt_simple.py` (task has neither
`[[shared]]` entry for it nor `[variant]` block and no `--variant` was passed),
runner raises `FileNotFoundError` before dispatch.

### `--backend harbor` (containerized, cross-agent comparable)

Requires: `harbor` CLI, docker with `nvidia-container-runtime`, GPU.

Runs `harbor run -p <mounted_task> -a bash -n 1 --env docker --seed N`.
Harbor builds the task's Dockerfile against `<mounted_task>/environment/`
as build context — so shared assets injected by mount are picked up by Docker COPY.

Runner locates Harbor's `result.json` under `--jobs-dir`, finds the log in
`trial_dir/artifacts/runs/`, then re-grades locally for cross-check.

## Error and edge cases

| Situation | Runner behavior | Ledger row |
|---|---|---|
| Task name not found | `FileNotFoundError` before mount | not written |
| Task lacks required `--variant` | `ValueError` at mount | status=error, reason set |
| Task has no `[variant]` block but --variant passed | `ValueError` at mount | status=error, reason set |
| Log missing after dispatch | RuntimeError from dispatcher | status=error |
| Grader fails to parse | Grader emits `reward: 0.0` with `reason` | status=success, reward=0.0 |
| CUDA OOM (local backend) | nonzero returncode, log parsed anyway | status=error, partial metrics if log has checkpoints |
| Harbor build fails | RuntimeError from dispatch_harbor | status=error, harbor stderr in error field |

## What the orchestrator does between runs

1. Read tail of `runs.jsonl`, filter by `task_id` (one file, many tasks).
2. Update `plan.md` "Current state" for the relevant task.
3. Update `plan.md` picklist.
4. Log the why to `scratchpad/THREAD.md` with UTC timestamp.
5. Apply AGENTS.md gates (§Law 2 noise-floor, §Law 3 stuck-detector,
   §Law 5 2-seed reproduction, §Law 6 pruning) per task family.
