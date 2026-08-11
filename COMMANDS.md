# COMMANDS

Copy-paste recipes. Run everything from the repo root.

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

```bash
python -m pytest tests/ -v    # 38 tests, ~0.2s
```

Expected breakdown:
- 7 grader (task-specific reward computation)
- 4 harness_dry (multi-task orchestrator end-to-end)
- 4 ingest_result (canonical row schema)
- 6 mount_toml parametrized (shared refs + variant block validity)
- 8 mount_variant (mount + swap + reject-invalid)
- 9 task_toml parametrized (per-task schema + required files + multi-task assertion)

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
python runner/mount_variant.py tasks/nanogpt-speedrun /tmp/mounted_task

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
python runner/mount_variant.py tasks/nanogpt-speedrun /tmp/mounted_task
harbor run -p /tmp/mounted_task -a oracle -n 1 --env docker
```

## Autonomous refinement loop (multi-iteration)

The runner supports LLM-driven refinement. Passing `--iterations N > 1` delegates
to `runner/orchestrator.run_loop`, which builds an LLM system prompt from the
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
  --agent claude_code \
  --iterations 20 --seeds 2
```

- SIGINT (Ctrl-C) halts cleanly after the current iteration finishes.
- Restart resumes from `max(iter*.py) + 1` — variant files under
  `scratchpad/variants/` are the recovery source of truth.
- The LLM never invokes the trainer itself; the outer runner does that after each
  `write_variant`.

## Inspect the ledger

```bash
# All rows
cat runs.jsonl

# Filter by task
python -c "
import json
for line in open('runs.jsonl'):
    r = json.loads(line)
    if r['task_id'] == 'nanogpt/track3-speedrun':
        print(f\"{r['variant']:30} seed={r['seed']} reward={r['reward']} step_to_3.28={r['step_to_3_28']}\")
"

# Multi-seed aggregation (outer loop's responsibility per AGENTS.md §Law 5)
python -c "
import json, collections, statistics
rows = [json.loads(l) for l in open('runs.jsonl')]
by_variant = collections.defaultdict(list)
for r in rows:
    if r['task_id'] == 'nanogpt/track3-speedrun' and r['hit_target']:
        by_variant[r['variant']].append(r['step_to_3_28'])
for v, steps in by_variant.items():
    print(f'{v:30} n={len(steps)} mean_steps={statistics.mean(steps):.1f} min={min(steps)}')
"
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

# Verify schema
python -m pytest tests/ -k my-task -v

# Smoke-test wiring
python runner/harness.py --task my-task --seeds 1 --backend dry
```

## Pruning round (AGENTS.md §Law 6) — nanogpt-speedrun

```bash
for v in loo_no_mod1 loo_no_mod2 loo_no_mod3; do
  python runner/harness.py \
    --task nanogpt-speedrun \
    --variant policy/scratchpad/variants/${v}.py \
    --seeds 1 --backend local
done
```

## Housekeeping

```bash
# Reset ledger (destructive)
mv runs.jsonl runs.jsonl.$(date +%Y%m%d%H%M%S).bak && touch runs.jsonl

# Clean run artifacts
rm -rf runs/*      # clear work dirs (keep the runs/ folder)
rm -f runs.jsonl   # clear the ledger

# Rebuild Harbor image (per-task; must mount first)
python runner/mount_variant.py tasks/nanogpt-speedrun /tmp/mounted
docker build -t nanogpt-track3:cuda126 /tmp/mounted/environment/
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: task not found` | Wrong `--task` name | `ls tasks/` — check spelling / path |
| `ValueError: task X does not accept a --variant` | Task's mount.toml has no `[variant]` | Drop `--variant` or add block |
| `ValueError: task X requires a --variant but none was provided` | `[variant].required=true` | Pass `--variant <path>` |
| `FileNotFoundError: shared asset missing` | mount.toml references file not in `shared/` | Add file or fix path |
| `FileNotFoundError: trainer missing at ...` (local backend) | Task didn't populate `environment/train_gpt_simple.py` | Add `[[shared]]` entry or pass `--variant` |
| `docker build tasks/nanogpt-speedrun/environment/` fails | Trainer only injected at mount time — not checked in | `python runner/mount_variant.py tasks/nanogpt-speedrun /tmp/mounted && docker build /tmp/mounted/environment/` |
| Reward=0.0 on nanogpt-smoke with `--backend dry` | Dry log format is Track-3-shaped; smoke grader looks for python+torch lines | Expected — dry proves wiring, not task-specific grading |
| `StopIteration` in `_load_data_shard` | CWD not the environment/ dir | Runner sets `cwd=env_dir`; if bypassing runner, `cd <mounted>/environment` first |
| `NaN` val_loss with `torch==2.10` on A100 | Known upstream bug | Use `torch==2.11` (pinned) |
| Empty ledger despite runs | Wrong `--ledger` path | Defaults to `runs.jsonl` at repo root; check `pwd` |
| `harbor run` produces no `result.json` | Harbor CLI not in PATH or version mismatch | `harbor --version`; pin v0.16.1+ |
| Pytest fails on new task | Missing required file | Verify all 7 files per §Adding a new task; run `pytest tests/test_task_toml.py -v` |

## Reference

- Pipeline flow: `FLOW.md`
- Agent conduct: `policy/AGENTS.md`
- Mission (Track-3): `policy/goal.md`
- Live state: `policy/plan.md`
- Track-3 spec mirror: `shared/track3_README.md`
