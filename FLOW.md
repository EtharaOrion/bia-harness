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
│       --seeds N --backend {harbor,local,dry}                            │
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
│  7. Ingest (runner/ingest_result.py)                                    │
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
