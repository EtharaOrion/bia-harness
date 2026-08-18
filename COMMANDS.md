# COMMANDS

Copy-paste recipes. Run everything from the repo root.

## agentloop refinement loop (primary path)

The agent authors `/workspace/submission/optimizer.py` and trains it inside the
container; this drives the outside of that loop. Use it for `bia/track3nov` tasks.

```bash
# every flag, for reference
python runner/agentloop/loop.py --help

# resume-or-start a campaign (the ledger decides which)
python runner/agentloop/loop.py --task 2739a678-1759-516d-8ba7-1cd023267ea8 --iterations 3

# force a specific iteration number
python runner/agentloop/loop.py --task <name|uuid|path> --iterations 1 --start-at 7

# pick the in-container agent (both use the same Claude OAuth bridge)
python runner/agentloop/loop.py --task <task> --iterations 2 --agent openhands-sdk

# bound each harbor job; a job killed by this records harbor_returncode=124.
# NO limit by default, so a wedged container hangs the campaign forever without it.
python runner/agentloop/loop.py --task <task> --iterations 3 --timeout 14400

# keep only the newest 2 job dirs under runs/agentloop/<slug>/jobs after each
# successful iteration. Unset = keep everything. Never deletes the dir just produced.
python runner/agentloop/loop.py --task <task> --iterations 5 --keep-jobs 2

# opt in to the LLM judge / summariser (both EXPERIMENTAL and unwired by default)
python runner/agentloop/loop.py --task <task> --iterations 2 --summarise --judge

# point at a specific harbor build
HARBOR_BIN=/home/bia-gpu/oer/.venv-harbor/bin/harbor \
  python runner/agentloop/loop.py --task <task> --iterations 1
# or explicitly
python runner/agentloop/loop.py --task <task> --iterations 1 \
  --harbor-bin /home/bia-gpu/oer/.venv-harbor/bin/harbor
```

One campaign per task at a time: `refine` holds an exclusive `flock` on
`runs/agentloop/<slug>/.lock` for its whole duration and a second run raises
`RunRootBusy` rather than interleaving into the first one's ledger and iteration
numbering. There is no `--force`. If a lock looks stuck, it is not — flock is released
by the kernel when the holder dies — so find the live holder:

```bash
fuser runs/agentloop/<slug>/.lock
```

Inspect a campaign:

```bash
cat runs/agentloop/<slug>/ledger.jsonl | python -m json.tool --json-lines
cat runs/agentloop/<slug>/history/iter02_history.md      # what the agent was told
cat runs/agentloop/<slug>/history/iter02_facts.json      # facts, before LLM enrichment
cat runs/agentloop/<slug>/.cfg_iter02.json               # exact harbor config used
```

Opt-in integration test (launches a real container, ~20s):

```bash
TRACK3_INTEGRATION=1 pytest tests/test_agentloop_integration.py -v
```

Caveat: this loop has only ever been driven with harbor's `nop` agent. It has never
been run with a real LLM agent end to end.

## Setup

### CPU host (wiring tests only)

```bash
python -m pip install pytest
```

### GPU host (real runs)

```bash
python -m pip install -r requirements.txt

# One-time: cache FineWeb-10B shards (nanogpt-speedrun task)
python shared/data/cached_fineweb10B.py 20    # 2B tokens, ~4000 steps worth
```

### GPU host with Harbor

```bash
pip install harbor-cli
docker version                # verify docker + nvidia-container-runtime
harbor --version              # verify CLI in PATH
```

## Verify the harness

`pyproject.toml` sets `testpaths = ["tests", "legacy/harness2/tests"]`, so the suite
command is a **bare `pytest`** with no path argument. Passing a path overrides
`testpaths` and silently collects only half the suite.

```bash
# the whole suite, both trees
pytest

# the new pipeline only (agentloop + the shared harness) — fully green
pytest tests/

# the legacy planner substrate only
pytest legacy/harness2/tests
```

At the time of writing: `pytest` gives `2 failed, 640 passed, 10 skipped`;
`pytest tests/` gives `437 passed, 2 skipped`; `pytest legacy/harness2/tests` gives
`2 failed, 203 passed, 8 skipped`. The two failures are both
`legacy/harness2/tests/test_task_toml.py::test_task_toml_parses` against the two
`bia/track3nov` bundles, which declare `schema_version = "1.4"` where that test asserts
`version = "1.0"`. They are pre-existing and have nothing to do with the restructure.

The suite covers task grading, dry-run orchestration, ledger ingestion, mount
validation, task schemas, LLM client behavior, RFP/CLI alignment, and the whole
agentloop pipeline. Do not copy a test count into automation; the count changes as
coverage grows.

### Verify a run with deterministic pytest and the GPT judge

`bia_verifier` moved to `legacy/harness2/bia_verifier/` with the rest of the legacy
planner path. It is not invoked by `agentloop` or by any current run path; it is kept
because a verifier may be rewired into the loop later. Invoke it by its new module
path, from the repo root:

```bash
export KAIJU_CODEX_BRIDGE_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export PYTHONPATH="$PWD/codex-proxy"
python -m agent.openai_codex --host 127.0.0.1 --port 8788

# In a second shell, using the same secret:
export OPENAI_BASE_URL=http://127.0.0.1:8788
export OPENAI_API_KEY="$KAIJU_CODEX_BRIDGE_SECRET"
python -m legacy.harness2.bia_verifier.cli grade \
  --dataset path/to/dataset.json \
  --run path/to/run \
  --predicates path/to/predicates.py \
  --rubric path/to/rubric.json \
  --judge codex --judge-model gpt5.6-sol \
  --out output/run-id
```

See `personal-docs/BIA_VERIFIER.md` for schemas, trust boundaries, offline judgement
replay, artifact definitions, and operational cautions.

## Run tasks

### List available tasks

```bash
ls tasks/
```

### Task 1: nanogpt-speedrun (with a candidate variant)

```bash
cp shared/train_gpt_simple.py policy/scratchpad/variants/my_variant.py
# edit my_variant.py — modify only Optimization + Init & Optim Hyperparams

python runner/harness.py \
  --task nanogpt-speedrun \
  --variant policy/scratchpad/variants/my_variant.py \
  --seeds 2 --backend local
```

### Task 2: nanogpt-smoke (no variant, no shared assets)

```bash
python runner/harness.py \
  --task nanogpt-smoke \
  --seeds 1 --backend local
```

### Any task via path (not registered under tasks/)

```bash
python runner/harness.py \
  --task /path/to/my-external-task \
  --variant policy/scratchpad/variants/some.py \
  --seeds 1 --backend local
```

### Harbor backend

```bash
python runner/harness.py \
  --task nanogpt-speedrun \
  --variant policy/scratchpad/variants/my_variant.py \
  --seeds 2 --backend harbor \
  --out-root /tmp/harbor_jobs
```

### Direct Harbor CLI (bypass runner, no variant swap)

```bash
# First mount the task (populates environment/train_gpt_simple.py from shared/)
python legacy/harness2/mount_variant.py tasks/nanogpt-speedrun /tmp/mounted_task

# Then run Harbor against the mounted directory
harbor run \
  -p /tmp/mounted_task \
  -a bash \
  -n 2 -j 2 \
  --env docker \
  --jobs-dir /tmp/harbor_jobs
```

### Oracle (calibrate wallclock + verify plumbing)

```bash
python legacy/harness2/mount_variant.py tasks/nanogpt-speedrun /tmp/mounted_task
harbor run -p /tmp/mounted_task -a oracle -n 1 --env docker
```

## Autonomous refinement loop (multiple attempts)

The runner supports LLM-driven refinement. Passing `--attempts N > 1` delegates
to `legacy/harness2/orchestrator.run_loop`, which builds an LLM system prompt from the
shared `policy/AGENTS.md` + the task's own `instruction.md` + the auto-refreshed
`policy/<slug>/plan.md` + `goal.md`, calls the LLM with tool_use tools
(`write_variant`, `append_thread`, `update_plan_section`, `add_ruled_down`),
executes the tool calls (writes `iter<N>.py` under `scratchpad/variants/`),
then invokes the harness with the new variant, appends the result to
`runs/<slug>/runs.jsonl`, and loops.

First-time invocation auto-scaffolds `policy/<slug>/` from the task's own
`task.toml [task].description` + `instruction.md` — no hand-authored files.

```bash
python runner/harness.py \
  --task bia-track3-optimizer-novelty \
  --backend harbor \
  --llm-config proxy/claude-code-oauth.json \
  --agent claude-code \
  --attempts 20 --seeds 2 --llm-retries 3
```

- `--attempts` is the canonical RFP term. `--iterations` remains a deprecated
  alias for compatibility.
- `--seeds` defaults to 2 and controls independent RNG replicas per attempt.
- `--llm-retries` defaults to 3 and retries only planner LLM transport failures
  such as timeouts and HTTP 429 responses. It does not repeat training seeds,
  retry failed Harbor jobs, or add attempts.
- The retired `--retries` flag is rejected. Use `--llm-retries` for planner API
  retries and `--seeds` for statistical replication.
- SIGINT (Ctrl-C) halts cleanly after the current iteration finishes.
- Restart resumes from `max(iter*.py) + 1` — variant files under
  `scratchpad/variants/` are the recovery source of truth.
- The LLM never invokes the trainer itself; the outer runner does that after each
  `write_variant`.

## Inspect the ledger

The legacy path's ledger is per task: `runs/<task-slug>/runs.jsonl`, where the slug is
`[task].name` from `task.toml` with `/` replaced. `nanogpt/track3-speedrun` writes to
`runs/nanogpt-track3-speedrun/runs.jsonl`; `nanogpt/env-smoke` to
`runs/nanogpt-env-smoke/runs.jsonl`. Override with `--ledger PATH`. (The agentloop path
keeps its own separate ledger at `runs/agentloop/<slug>/ledger.jsonl`.)

```bash
# What ledgers exist
ls runs/*/runs.jsonl

# All rows for one task
LEDGER=runs/nanogpt-track3-speedrun/runs.jsonl
cat "$LEDGER"

# Per-row digest
python -c "
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    print(f\"{r['variant']:30} seed={r['seed']} reward={r['reward']} step_to_3.28={r['step_to_3_28']}\")
" "$LEDGER"

# Multi-seed aggregation (outer loop's responsibility per AGENTS.md §Law 5)
python -c "
import json, collections, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
by_variant = collections.defaultdict(list)
for r in rows:
    if r['hit_target']:
        by_variant[r['variant']].append(r['step_to_3_28'])
for v, steps in by_variant.items():
    print(f'{v:30} n={len(steps)} mean_steps={statistics.mean(steps):.1f} min={min(steps)}')
" "$LEDGER"
```

## Adding a new task

```bash
mkdir -p tasks/my-task/{environment,solution,tests}

# 1. task.toml — Harbor schema (see tasks/nanogpt-smoke/task.toml for minimal)
# 2. mount.toml — optional; declares shared assets + variant target
# 3. instruction.md
# 4. environment/Dockerfile
# 5. solution/solve.sh (chmod +x)
# 6. tests/test.sh (chmod +x, Harbor verifier entrypoint)
# 7. tests/grader.py — stdlib-executable; emits JSON {reward: float, ...}

chmod +x tasks/my-task/solution/solve.sh tasks/my-task/tests/test.sh

# Verify schema — bare pytest, so testpaths pick up legacy/harness2/tests too
pytest -k my-task -v

# Smoke-test wiring
python runner/harness.py --task my-task --seeds 1 --backend dry
```

## Pruning round (AGENTS.md §Law 6) — nanogpt-speedrun

```bash
for v in loo_no_mod1 loo_no_mod2 loo_no_mod3; do
  python runner/harness.py \
    --task nanogpt-speedrun \
    --variant policy/nanogpt-track3-speedrun/scratchpad/variants/${v}.py \
    --seeds 1 --backend local
done
```

## Housekeeping

```bash
# Reset one task's legacy ledger (destructive)
LEDGER=runs/nanogpt-track3-speedrun/runs.jsonl
mv "$LEDGER" "$LEDGER.$(date +%Y%m%d%H%M%S).bak" && touch "$LEDGER"

# Clean run artifacts (destructive; also removes every ledger under runs/)
rm -rf runs/*      # clear work dirs (keep the runs/ folder)

# Prune old agentloop job dirs without touching the ledger — prefer --keep-jobs N
# on the loop itself, which never deletes the dir the current iteration produced.

# Rebuild Harbor image (per-task; must mount first)
python legacy/harness2/mount_variant.py tasks/nanogpt-speedrun /tmp/mounted
docker build -t nanogpt-agentloop:cuda126 /tmp/mounted/environment/
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: task not found` | Wrong `--task` name | `ls tasks/` — check spelling / path |
| `ValueError: task X does not accept a --variant` | Task's mount.toml has no `[variant]` | Drop `--variant` or add block |
| `ValueError: task X requires a --variant but none was provided` | `[variant].required=true` | Pass `--variant <path>` |
| `FileNotFoundError: shared asset missing` | mount.toml references file not in `shared/` | Add file or fix path |
| `FileNotFoundError: trainer missing at ...` (local backend) | Task didn't populate `environment/train_gpt_simple.py` | Add `[[shared]]` entry or pass `--variant` |
| `docker build tasks/nanogpt-speedrun/environment/` fails | Trainer only injected at mount time — not checked in | `python legacy/harness2/mount_variant.py tasks/nanogpt-speedrun /tmp/mounted && docker build /tmp/mounted/environment/` |
| Reward=0.0 on nanogpt-smoke with `--backend dry` | Dry log format is Track-3-shaped; smoke grader looks for python+torch lines | Expected — dry proves wiring, not task-specific grading |
| `StopIteration` in `_load_data_shard` | CWD not the environment/ dir | Runner sets `cwd=env_dir`; if bypassing runner, `cd <mounted>/environment` first |
| `NaN` val_loss with `torch==2.10` on A100 | Known upstream bug | Use `torch==2.11` (pinned) |
| Empty ledger despite runs | Looking in the wrong place | The ledger is per task: `runs/<task-slug>/runs.jsonl`. `ls runs/*/runs.jsonl` |
| `unrecognized arguments: --retries` | Retired ambiguous flag | Use `--llm-retries N` for planner retries, `--retry N` for infra dispatch retries, `--seeds N` for replicas |
| `--attempts > 1 requires --llm-config` | Autonomous mode needs an LLM endpoint | Pass a proxy JSON via `--llm-config` or run a single manual attempt |
| `harbor run` produces no `result.json` | Harbor CLI not in PATH or version mismatch | `harbor --version`; pin v0.16.1+ |
| Pytest fails on new task | Missing required file | Verify all 7 files per §Adding a new task; run `pytest legacy/harness2/tests/test_task_toml.py -v` |
| `pytest tests/` passes but `pytest` fails | The two trees are different halves of one suite | `testpaths` only applies to a bare `pytest`; a path argument overrides it |
| `RunRootBusy: another refine run is already active` | A second agentloop campaign on the same task | Wait for it; `fuser runs/agentloop/<slug>/.lock` names the holder. There is no `--force` |
| `[warn] ... unreadable ledger line SKIPPED` | A kill mid-append truncated a line | Inspect the file before continuing — `start = len(rows)+1`, so a dropped row renumbers every iteration after it |
| `harbor_returncode: 124` in a ledger row | `--timeout` killed that job | Raise `--timeout`, or investigate why the container wedged |
| `harbor_returncode: 130` plus `interrupted: true` | Ctrl-C during that job | Expected. The row records what was measured before the interrupt; re-run to resume |
| `ModuleNotFoundError: bia_verifier` | It moved under the legacy path | `python -m legacy.harness2.bia_verifier.cli ...` from the repo root |

## Reference

- Pipeline flow: `FLOW.md`
- Superseded planner path (data-flow diagram): `DFD.md`
- Agent conduct: `policy/AGENTS.md`
- Mission + live state, per task: `policy/<task-slug>/{goal.md, plan.md}`
- Track-3 spec mirror: `shared/track3_README.md`
