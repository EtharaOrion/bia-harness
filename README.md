# bia-harness

Multi-task Harbor-compatible harness for autonomous LLM-driven ML research.
Bundles Prime Intellect's markdown-driven agent policy with a task registry
(`tasks/*`) where each subdir is a self-contained Harbor task. The included
Track-3 optimization task is one example; add more by dropping a directory
under `tasks/`.

## Layout

```
.
├── policy/                              markdown-only agent policy (task-agnostic)
│   ├── AGENTS.md, goal.md, plan.md, README.md
│   └── scratchpad/                      THREAD.md, picklist.md, audits.md, variants/, ideas/, papers/
├── shared/                              cross-task assets injected at mount time
│   ├── train_gpt_simple.py              Track-3 baseline trainer (used by nanogpt-speedrun)
│   ├── track3_README.md                 official spec mirror
│   └── data/cached_fineweb10B.py        FineWeb-10B downloader
├── tasks/                               task registry
│   ├── nanogpt-speedrun/                the Track-3 speedrun task
│   │   ├── task.toml                    Harbor v1.0 (name='nanogpt/track3-speedrun')
│   │   ├── mount.toml                   [[shared]] + [variant]
│   │   ├── instruction.md
│   │   ├── environment/Dockerfile       (trainer injected at mount time)
│   │   ├── solution/solve.sh
│   │   └── tests/{test.sh, grader.py}
│   └── nanogpt-smoke/                   env-sanity task (no shared, no variant)
│       ├── task.toml (name='nanogpt/env-smoke', gpus=0)
│       ├── mount.toml                   version-only (empty)
│       ├── instruction.md
│       ├── environment/Dockerfile       (python:3.12-slim + torch cpu)
│       ├── solution/solve.sh
│       └── tests/{test.sh, grader.py}
├── runner/                              multi-task orchestrator
│   ├── harness.py                       --task <name|path> --variant <path> --backend {harbor,local,dry}
│   ├── mount_variant.py                 mount_task(task, out, variant) reads mount.toml
│   └── ingest_result.py                 20-field canonical row -> runs.jsonl
├── tests/                               pytest (38 tests, parametrized over tasks/*)
├── records/                             upstream Track-3 reference (results history, baselines)
├── legacy/                              Track-1 substrate + prior audit docs (not on run path)
├── runs.jsonl                           append-only ledger (task_id per row)
├── LICENSE, pyproject.toml, requirements.txt
└── README.md, FLOW.md, COMMANDS.md
```

## Design principles

1. **Task-agnostic core**. Runner + mount + ingest know nothing about
   nanogpt. Add a new task by dropping a directory under `tasks/`.
2. **Shared assets outside tasks**. Large/duplicated files (trainers, data
   downloaders) live in `shared/`, injected into the mounted task per each
   task's `mount.toml`.
3. **Task-specific config in `task.toml` + `mount.toml`**. `task.toml` is
   Harbor's schema (docker image, timeouts, GPU); `mount.toml` declares
   which shared assets the task needs and where `--variant` lands.
4. **Policy is task-agnostic**. `policy/AGENTS.md` describes the *conduct*
   of an autonomous optimization loop; `policy/goal.md` frames the specific
   mission. Swap `goal.md` for a different mission, keep `AGENTS.md`.

## Two things, cleanly separated

- **Policy** (markdown, in `policy/`) — the conduct spec the orchestrator
  reads at session start.
- **Execution** (task + runner) — the reward machine. Runner mounts, dispatches,
  grades, appends structured results to `runs.jsonl`.

Autonomous loop: orchestrator writes `policy/scratchpad/variants/<slug>.py`
per `AGENTS.md` rules → runner mounts variant into the chosen task → dispatch
to {harbor,local,dry} → grader emits `reward.json` → ingest normalizes into
`runs.jsonl` → orchestrator reads the ledger and updates `plan.md`.

## Quickstart

```bash
python -m pytest tests/

# Task 1 with variant swap
python runner/harness.py --task nanogpt-speedrun \
  --variant policy/scratchpad/variants/demo_smoke.py \
  --seeds 2 --backend dry

# Task 2 without variant (env sanity)
python runner/harness.py --task nanogpt-smoke \
  --seeds 1 --backend dry

# Work artifacts land under runs/<task>_<variant>_seed<N>_<utc-stamp>/
# Ledger rows land in runs.jsonl (append-only).
ls runs/
cat runs.jsonl
```

For real GPU runs, see `COMMANDS.md`. Full stage-by-stage pipeline in `FLOW.md`.
Verifier setup, rubric authoring, Codex judge operation, pytest enforcement, and
artifact contracts are documented in `BIA_VERIFIER.md`.

For autonomous refinement, use `--attempts N` (`--iterations` is a deprecated
alias). Each attempt evaluates one LLM-authored candidate across `--seeds S`,
which defaults to 2. `--llm-retries R`, default 3, applies only to planner LLM
429/timeout retries; it does not retry training runs. The old ambiguous
`--retries` flag is intentionally rejected.

## Adding a new task

1. Create `tasks/<my-task>/` with: `task.toml`, `mount.toml`, `instruction.md`,
   `environment/Dockerfile`, `solution/solve.sh`, `tests/{test.sh, grader.py}`.
2. If your task needs assets from `shared/`, declare them in `[[shared]]`
   blocks of `mount.toml`. Optional `[variant]` block declares where a
   `--variant` file lands.
3. Your `tests/grader.py` must be stdlib-executable and emit
   `{reward: 0.0-1.0, ...metrics}` JSON (to stdout for the runner path, or
   to `/logs/verifier/reward.json` for the Harbor `test.sh` path).
4. Smoke-test: `python runner/harness.py --task <my-task> --seeds 1 --backend dry`.
5. `python -m pytest tests/` automatically parametrizes over all tasks and
   validates `task.toml` + `mount.toml` for your new task.

## Not on this machine

CPU host: scaffold + logic verified via `--backend dry`. GPU host: use
`--backend local` (bare torchrun) or `--backend harbor` (containerized trial).

## Provenance

Policy merged from Prime Intellect's `experiments-autonomous-speedrunning`;
Track-3 substrate from `KellerJordan/modded-nanogpt`. See `policy/README.md`
for full lineage. Historical audit docs and Track-1 code preserved under
`legacy/`.
