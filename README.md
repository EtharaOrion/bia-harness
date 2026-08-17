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
│   ├── ingest_result.py                 20-field canonical row -> runs.jsonl
│   └── track3/                          code-driven refinement loop (agent authors code in-container)
│       ├── loop.py                      CLI + refine(); the ledger is the loop state
│       ├── harbor_config.py             builds/validates the harbor --config JSON
│       ├── history.py                   renders prior iterations into agent-facing markdown
│       ├── trial_io.py                  one trial dir -> one flat ledger row
│       └── classify.py, judge.py, summariser.py, marking.py
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

## Two refinement loops, and which one a task can use

- **`runner/orchestrator.run_loop`** (via `harness.py --attempts N`). A planner LLM
  authors a candidate *outside* the container and the harness injects it as a
  `--variant` at mount time. This requires the task to declare a `[variant]` block
  in `mount.toml`, so it **cannot** work for a task whose `/workspace` ships inside
  the docker image — there is nothing to mount a variant into.
- **`runner/track3`**. The agent authors `/workspace/submission/optimizer.py` and
  runs training *inside* the container; our code drives the outside of that loop.
  Each iteration renders prior results into markdown, injects it through
  `extra_instruction_paths` in a harbor `--config` JSON, launches the job, locates
  and parses the resulting trial, classifies the outcome, and appends one ledger row.

Use `run_loop` when the harness owns the candidate; use `track3` when the container
owns it.

```bash
python runner/track3/loop.py --task <name|uuid|path> --iterations N \
  [--start-at N] [--summarise] [--judge] [--harbor-bin PATH]
```

On disk, per task:

```
runs/track3/<slug>/
├── ledger.jsonl                 one JSON row per iteration (append-only)
├── history/iterNN_history.md    the markdown injected into iteration NN
├── history/iterNN_facts.json    measured facts, checkpointed before LLM enrichment
├── .cfg_iterNN.json             the exact config handed to `harbor run`
└── jobs/<job_name>/<trial>/     harbor's own trial output
```

`<slug>` comes from the task's uuid, falling back to a slug of `[task].name`.

### Reward

`runner/track3/reward.py` scores an iteration from the steps it took to reach the
target loss:

```
reward = max(0.0, min(1.0, (BASELINE_STEPS - step) / (BASELINE_STEPS - TARGET_STEPS)))

BASELINE_STEPS = 3500    TARGET_STEPS = 2900    TARGET_LOSS = 3.28
```

`step` is the first point at which **every** seed has reached `TARGET_LOSS`, so a run
is only as fast as its slowest seed. Reaching target at 3500 scores 0.0, at 3200
scores 0.5, at 2900 or better scores 1.0. A run that never reaches target scores 0.0.

The step is measured on the **full-density** log points, not on the thinned
`parent_curve` that gets rendered into the prompt — thinning is a display budget and
must never move a score.

### Currently unwired

The task's own verifier and the LLM judge/summariser are **not** in the default path.
Reward comes from the formula above; `judge.py` and `summariser.py` remain on disk and
tested, and are opt-in via `--judge` / `--summarise`.

**The ledger is the loop state.** `start` is derived from its length and every row is
appended the moment it is produced, so an interrupted campaign resumes where it
stopped instead of restarting — just re-run the same command. `--start-at` overrides
that only when you need it.

**`--export-traces` is passed as a CLI flag, never as a config key.** Harbor's
`JobConfig` is pydantic `extra="ignore"`, so an unknown key is silently dropped: an
`export_traces` entry in the JSON would leave the config looking correct while harbor
wrote no `agent/trajectory.json`, and every seed would come back
`verification_incomplete`. For the same reason `harbor_config.validate_cfg` is
deliberately stricter than harbor and rejects unknown keys at every depth.

**Synthetic results are marked and bannered.** The `--backend dry` path fabricates a
val_loss curve from a seed hash — no model, no data, no gradients. Those rows carry
`is_synthetic` and any surface rendering them (including the agent-facing history)
prints a loud banner above the facts table. They must never be reported as real
training results.

The real harbor/docker path is covered by `tests/test_track3_integration.py`, which is
opt-in: it only runs under `TRACK3_INTEGRATION=1` with docker and the task image
present, because one iteration starts a real GPU container.

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
