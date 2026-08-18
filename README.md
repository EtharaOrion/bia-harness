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
│   ├── AGENTS.md, README.md             the conduct spec, shared by every task
│   └── <task-slug>/                     scaffolded per task on first autonomous run
│       ├── goal.md, plan.md             mission + live state
│       ├── scratchpad/                  THREAD.md, picklist.md, audits.md, variants/
│       └── runNN/                       frozen {goal.md, plan.md, variants/} per attempt
├── shared/                              cross-task assets injected at mount time
│   ├── train_gpt_simple.py              Track-3 baseline trainer (used by nanogpt-speedrun)
│   ├── track3_README.md                 official spec mirror
│   └── data/cached_fineweb10B.py        FineWeb-10B downloader
├── tasks/                               task registry
│   ├── nanogpt-speedrun/                the Track-3 speedrun task
│   │   ├── task.toml                    version='1.0' (name='nanogpt/track3-speedrun')
│   │   ├── mount.toml                   [[shared]] + [variant]
│   │   ├── instruction.md
│   │   ├── environment/Dockerfile       (trainer injected at mount time)
│   │   ├── solution/solve.sh
│   │   ├── tests/{test.sh, grader.py}
│   │   └── verifier/                    dataset/predicates/rubric for the legacy verifier
│   ├── nanogpt-smoke/                   env-sanity task (no shared, no variant)
│   │   ├── task.toml (name='nanogpt/env-smoke', gpus=0)
│   │   ├── mount.toml                   version-only (empty)
│   │   ├── instruction.md
│   │   ├── environment/Dockerfile       (python:3.12-slim + torch cpu)
│   │   ├── solution/solve.sh
│   │   └── tests/{test.sh, grader.py}
│   └── <uuid>/                          two bia/track3nov bundles, schema_version='1.4';
│                                        /workspace ships in the image, so no mount.toml
│                                        and no [variant] — agentloop serves these
├── runner/                              the CURRENT pipeline, and nothing else
│   ├── __init__.py
│   ├── harness.py                       --task <name|path> --variant <path> --backend {harbor,local,dry}
│   │                                    SHARED. agentloop imports exactly four symbols from it —
│   │                                    HARNESS_ROOT, resolve_task, resolve_task_uuid,
│   │                                    resolve_harbor_bin — and never calls harness.run
│   │                                    or any dispatcher.
│   └── agentloop/                       code-driven refinement loop (agent authors code in-container)
│       ├── loop.py                      driver + CLI; refine(); the ledger is the loop state
│       ├── history.py                   renders prior iterations into agent-facing markdown,
│       │                                through the scrub firewall
│       ├── trial_io.py                  parse one completed trial -> one flat ledger row
│       ├── reward.py                    scores an iteration from the val_loss curve
│       ├── classify.py                  six outcomes
│       ├── harbor_config.py             builds/validates the harbor --config JSON
│       ├── marking.py                   synthetic-result marking + banner
│       └── judge.py, summariser.py      present and tested, but UNWIRED (see below)
├── tests/                               pytest for the current pipeline (agentloop + shared harness)
│   └── fixtures/track3_trial/           a REAL captured Harbor trial; 5 test files parse it
├── records/                             upstream Track-3 reference (results history, baselines)
├── legacy/                              off the current run path
│   ├── harness1/                        Track-1 substrate + prior audit docs (was the top-level legacy/)
│   └── harness2/                        planner-authors-the-variant loop (was under runner/)
│       ├── orchestrator.py              run_loop(): the host-side feedback loop
│       ├── mount_variant.py             mount_task(task, out, variant) reads mount.toml
│       ├── ingest_result.py             canonical row -> runs/<task-slug>/runs.jsonl
│       ├── llm_client.py, plan_writer.py, scaffold_policy.py, summarize.py
│       ├── bia_verifier/                moved here from the repo root; NOT used by agentloop
│       └── tests/                       its own pytest suite, moved with it
├── runs/                                run artifacts (gitignored)
│   ├── agentloop/<slug>/                agentloop campaigns
│   └── <task-slug>/runs.jsonl           the legacy path's per-task append-only ledger
├── personal-docs/                       the author's own notes, incl. BIA_VERIFIER.md (gitignored)
├── LICENSE, pyproject.toml, requirements.txt
└── README.md, FLOW.md, COMMANDS.md, DFD.md
```

`bia_verifier/` moved to `legacy/harness2/bia_verifier/` because it belongs to the
legacy path, not to `agentloop`. It was moved rather than deleted: a verifier may be
rewired into the loop later. It is importable as
`legacy.harness2.bia_verifier` from the repo root.

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
  grades, appends structured results to the task's ledger.

Autonomous loop: orchestrator writes
`policy/<task-slug>/scratchpad/variants/<slug>.py` per `AGENTS.md` rules → runner
mounts variant into the chosen task → dispatch to {harbor,local,dry} → grader emits
`reward.json` → ingest normalizes into `runs/<task-slug>/runs.jsonl` → orchestrator
reads the ledger and updates `plan.md`.

## Two refinement loops, and which one a task can use

- **`legacy/harness2/orchestrator.run_loop`** (via `harness.py --attempts N`).
  A planner LLM authors a candidate *outside* the container and the harness injects
  it as a `--variant` at mount time. This requires the task to declare a `[variant]`
  block in `mount.toml`, so it **cannot** work for a task whose `/workspace` ships
  inside the docker image — there is nothing to mount a variant into. Attempting it
  raises `harness.VariantDeliveryImpossible`. Serves `nanogpt-speedrun`.
- **`runner/agentloop`**. The agent authors `/workspace/submission/optimizer.py` and
  runs training *inside* the container; our code drives the outside of that loop.
  Each iteration renders prior results into markdown, injects it through
  `extra_instruction_paths` in a harbor `--config` JSON, launches the job, locates
  and parses the resulting trial, classifies the outcome, and appends one ledger row.

Use `run_loop` when the harness owns the candidate; use `agentloop` when the container
owns it. The two bia/track3nov bundles under `tasks/<uuid>/` ship `/workspace` inside
the image, so they can only be served by `agentloop`.

```bash
python runner/agentloop/loop.py --task <name|uuid|path> --iterations N \
  [--start-at N] [--agent claude-code|openhands-sdk] \
  [--summarise] [--judge] [--harbor-bin PATH] \
  [--timeout SECONDS] [--keep-jobs N]
```

**Not yet proven end to end.** Every run of this loop so far has used harbor's `nop`
agent, which starts the container and does nothing. The loop has NEVER been run with a
real LLM agent authoring an optimizer end to end, and `runs/agentloop/` on this machine
is empty. What is proven is the plumbing: config build/validation, launch, trial
location and parsing, scoring, classification and the ledger.

### Which agent runs in the container

`--agent` selects the harbor agent that authors the optimizer. Both reach the **same**
Claude OAuth bridge; only the env var names differ, because that is what each agent
reads:

| `--agent` | env it reads | model string |
|---|---|---|
| `claude-code` (default) | `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` | `claude-opus-5` |
| `openhands-sdk` | `LLM_BASE_URL`, `LLM_API_KEY` | `anthropic/claude-opus-5` |

`openhands-sdk` drives LiteLLM, which needs the `anthropic/` provider prefix to route
to the Messages API the bridge serves; the `claude` CLI wants the model bare. The
bridge itself is unchanged either way.

`openhands-sdk` pip-installs into `/opt/openhands-sdk-venv` during agent setup, which
is slower than claude-code's `npm install`. If setup times out, raise
`build_base_cfg(..., setup_timeout_multiplier=...)` (default `5.0`).

On disk, per task:

```
runs/agentloop/<slug>/
├── ledger.jsonl                 one JSON row per iteration (append-only)
├── history/iterNN_history.md    the markdown injected into iteration NN
├── history/iterNN_facts.json    measured facts, checkpointed before LLM enrichment
├── .cfg_iterNN.json             the exact config handed to `harbor run`
└── jobs/<job_name>/<trial>/     harbor's own trial output
```

`<slug>` comes from the task's uuid, falling back to a slug of `[task].name`.

### Reward

`runner/agentloop/reward.py` scores an iteration from the steps it took to reach the
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

A test parses the constants straight out of `tasks/<uuid>/tests/grade.py` and fails
loudly if the two ever drift apart: `reward.py` mirrors those numbers rather than
importing them, because `grade.py` is stdlib-only code that runs in the container.

### Currently unwired

The task's own verifier and the LLM judge/summariser are **not** in the default path.
Reward comes from the formula above; `judge.py` and `summariser.py` remain on disk and
tested, and are opt-in via `--judge` / `--summarise`. Both flags describe themselves as
EXPERIMENTAL in `--help`; a bare run reaches no LLM at all.

**The ledger is the loop state.** `start` is derived from its length and every row is
appended the moment it is produced, so an interrupted campaign resumes where it
stopped instead of restarting — just re-run the same command. `--start-at` overrides
that only when you need it.

### What the loop does when things go wrong

- **`--timeout SECONDS` bounds one harbor job.** It is threaded to `subprocess.run`;
  a job killed by it is recorded with `harbor_returncode=124` (GNU `timeout(1)`'s
  convention) and the campaign continues. Default is no limit, so a wedged container
  hangs the campaign forever unless you pass this.
- **Ctrl-C records before it re-raises.** An interrupt during a job writes the row for
  what was already measured, with `interrupted: true` and `harbor_returncode=130`, then
  lets the `KeyboardInterrupt` continue upward. An iteration that consumed GPU time
  always leaves a trace.
- **Containers created during an iteration are removed on exit.** Harbor's containers
  are not children of the harbor process, so a timeout or a Ctrl-C would leave them
  holding GPUs. Each iteration snapshots the running `task__*` containers before
  launching and force-removes only the *difference* in a `finally` — never a container
  that was already running, which on a shared host belongs to somebody else.
- **One campaign per task, enforced by `flock`.** `refine` runs under an exclusive
  lock on `run_root/.lock`. Two concurrent runs of one task would share the run root,
  the ledger, the `start` each derives from it, the job name and the config file; the
  second one raises `RunRootBusy` instead. There is deliberately no `--force`: flock is
  released by the kernel when the holder dies, so a held lock is always evidence of a
  live run. `fuser run_root/.lock` names the holder.
- **Ledger appends are one atomic write, under that lock.** A row can carry ~18000
  chars of `parent_source` and POSIX only guarantees atomic `O_APPEND` up to `PIPE_BUF`,
  so the whole line leaves in a single `os.write` followed by `fsync`.
- **`load_ledger` warns loudly instead of skipping in silence.** A malformed line is
  still skipped — rows are appended after every iteration, so a kill mid-write leaves a
  truncated final line and refusing to parse would strand every completed iteration
  before it. But each skip is printed with its line number and a summary, because
  `start = len(rows) + 1` means a dropped row silently renumbers every iteration after it.
- **`--keep-jobs N` prunes old job dirs, and is off by default.** After a successful
  iteration it deletes all but the newest N dirs under `run_root/jobs`, never the one
  just produced, never on the interrupted path, and never at all unless you pass the
  flag. Harbor trial artifacts are gigabytes per GPU trial; silently deleting one would
  be worse than the full disk it prevents.

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

The real harbor/docker path is covered by `tests/test_agentloop_integration.py`, which is
opt-in: it only runs under `TRACK3_INTEGRATION=1` with docker and the task image
present, because one iteration starts a real GPU container.

### `tests/fixtures/track3_trial/` — the tests parse real data, not mocks

Every other agentloop test replaces `subprocess.run` with a fake, but what that fake
materialises is a **real captured Harbor trial**, checked in at ~160K:

```
tests/fixtures/track3_trial/
├── config.json                              the job config harbor actually ran
├── result.json                              tokens, cost, timestamps, exception state
├── agent/{trajectory.json, claude-code.txt} the exported agent trace
├── verifier/{reward.json, reward_full.json} trial_io reads reward_full, not reward
└── artifacts/workspace/submission/
    ├── optimizer.py                         the optimizer the agent actually submitted
    └── logs/full_seed{0,1}.log              full-density val_loss curves, two seeds
```

Five test files parse it: `test_trial_io.py`, `test_agentloop_fixture.py`,
`test_agentloop_loop.py`, `test_agentloop_judge.py`, `test_agentloop_integration.py`.
That is the point — the parsers are exercised against a directory harbor genuinely
wrote, so a change in harbor's layout surfaces as a test failure rather than as a
reward-0 row in a live campaign.

The name keeps "track3" deliberately: this is a trial of the `bia/track3nov` **task**,
so the word describes where the data came from. It does not refer to the old
`runner/track3/` package name.

## Quickstart

```bash
# The suite is BOTH trees. pyproject sets testpaths = ["tests", "legacy/harness2/tests"],
# so the suite command is a BARE `pytest` with no path argument.
pytest

# The new-pipeline half only (agentloop + the shared harness) — fully green.
pytest tests/
```

As of this writing:

| command | result |
|---|---|
| `pytest` | 2 failed, 640 passed, 10 skipped |
| `pytest tests/` | 437 passed, 2 skipped |
| `pytest legacy/harness2/tests` | 2 failed, 203 passed, 8 skipped |

Both failures are `test_task_toml_parses` for the two `bia/track3nov` bundles, whose
`task.toml` declares `schema_version = "1.4"` while that test asserts `version = "1.0"`.
They are pre-existing and unrelated to the restructure; they are a stale assertion about
a newer Harbor schema, not a broken task.

```bash
# Task 1 with variant swap (any .py works; this one is a real prior candidate)
python runner/harness.py --task nanogpt-speedrun \
  --variant policy/nanogpt-track3-speedrun/scratchpad/variants/iter0.py \
  --seeds 2 --backend dry

# Task 2 without variant (env sanity)
python runner/harness.py --task nanogpt-smoke \
  --seeds 1 --backend dry

# Work artifacts land under runs/<task>_<variant>_seed<N>_<utc-stamp>/
# Ledger rows land in runs/<task-slug>/runs.jsonl (append-only), slug from task.toml
# [task].name — so nanogpt/env-smoke writes runs/nanogpt-env-smoke/runs.jsonl.
ls runs/
cat runs/nanogpt-env-smoke/runs.jsonl
```

For real GPU runs, see `COMMANDS.md`. Full stage-by-stage pipeline in `FLOW.md`. The
superseded planner path is diagrammed in `DFD.md`. Verifier setup, rubric authoring,
Codex judge operation, pytest enforcement, and artifact contracts are documented in
`personal-docs/BIA_VERIFIER.md` — all of which describe
`legacy/harness2/bia_verifier/`, which no current run path invokes.

For autonomous refinement, use `--attempts N` (`--iterations` is a deprecated
alias). Each attempt evaluates one LLM-authored candidate across `--seeds S`,
which defaults to 2. `--llm-retries R`, default 3, applies only to planner LLM
429/timeout retries; it does not retry training runs. The old ambiguous
`--retries` flag is intentionally rejected.

## Delivering a campaign

`tools/package_delivery.py` converts one agentloop campaign into the client delivery
format. See COMMANDS.md for flags and FLOW.md for the per-file transform table.

```bash
python tools/package_delivery.py --run-root runs/agentloop/<uuid> --out /tmp/delivery
```

The delivery carries the task bundle at its root and one `run_N` per iteration. Two
bundle-side files exist for that format rather than for the loop:

- `tasks/<task>/tests/emit_verifier_artifacts.py` writes `verifier/score.json`
  (numeric-only, the reason carried as an integer `reason_code`) and appends the full
  record to `grade-stdout.md`. Harbor parses every value in `score.json` as a number,
  so a string reason there drops the trial's score silently. `reward.json` and
  `reward_full.json` are still written; the loop reads `reward_full.json` for `reason`
  and `metrics.n_seeds`.
- `tasks/<task>/solution/provenance.yaml` records what the task was screened against
  and by what method. It states only screening actually performed and declares the rest
  absent with a reason, because a carrier describing screening that did not happen
  converts a visible gap into an invisible false claim.

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
5. A bare `pytest` automatically parametrizes over all tasks and validates
   `task.toml` + `mount.toml` for your new task. Those parametrized checks live in
   `legacy/harness2/tests/` and are only collected by the bare command, not by
   `pytest tests/`.

## Not on this machine

CPU host: scaffold + logic verified via `--backend dry`. GPU host: use
`--backend local` (bare torchrun) or `--backend harbor` (containerized trial).

## Provenance

Policy merged from Prime Intellect's `experiments-autonomous-speedrunning`;
Track-3 substrate from `KellerJordan/modded-nanogpt`. See `policy/README.md`
for full lineage. Historical audit docs and Track-1 code preserved under
`legacy/harness1/`; the superseded planner-authors-the-variant loop under
`legacy/harness2/`.
